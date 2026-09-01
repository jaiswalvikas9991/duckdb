#include "duckdb/optimizer/constant_or_null_simplification.hpp"

#include "duckdb/function/scalar/generic_common.hpp"
#include "duckdb/optimizer/expression_rewriter.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/expression_nullability.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"

namespace duckdb {

ConstantOrNullSimplification::ConstantOrNullSimplification(ClientContext &context_p) : context(context_p) {
}

static optional<bool> GetBooleanConstant(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return optional<bool>();
	}

	auto &constant = expr.Cast<BoundConstantExpression>().GetValue();
	if (constant.IsNull() || constant.type().id() != LogicalTypeId::BOOLEAN) {
		return optional<bool>();
	}

	return BooleanValue::Get(constant);
}

static optional<bool> GetConstantOrNullBoolean(Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION ||
	    expr.GetReturnType().id() != LogicalTypeId::BOOLEAN) {
		return optional<bool>();
	}

	auto &func = expr.Cast<BoundFunctionExpression>();
	if (ConstantOrNull::IsConstantOrNull(func, Value::BOOLEAN(true))) {
		return true;
	}

	if (ConstantOrNull::IsConstantOrNull(func, Value::BOOLEAN(false))) {
		return false;
	}

	return optional<bool>();
}

static bool ConstantOrNullInputsAreNotNull(LogicalOperator &input, BoundFunctionExpression &func,
                                           NotNullExpressionAnalyzer &analyzer) {
	auto &children = func.GetChildren();
	D_ASSERT(children.size() >= 2);

	for (idx_t child_idx = 1; child_idx < children.size(); ++child_idx) {
		if (children[child_idx]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			auto &constant = children[child_idx]->Cast<BoundConstantExpression>().GetValue();
			if (!constant.IsNull()) {
				continue;
			}
		}

		if (!analyzer.IsNotNull(input, *children[child_idx])) {
			return false;
		}
	}

	return true;
}

unique_ptr<Expression> ConstantOrNullSimplification::SimplifyExpression(LogicalOperator &input,
                                                                        unique_ptr<Expression> expr,
                                                                        NotNullExpressionAnalyzer &analyzer,
                                                                        bool allow_folding) {
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		child = SimplifyExpression(input, std::move(child), analyzer, allow_folding);
	});

	if (expr->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		// Folding constant_or_null into a plain constant drops the per-row evaluation of its
		// inputs, so it is only safe when the filter carries no volatile expressions.
		if (!allow_folding) {
			return expr;
		}
		auto value = GetConstantOrNullBoolean(*expr);
		if (!value.has_value()) {
			return expr;
		}

		auto &func = expr->Cast<BoundFunctionExpression>();
		if (ConstantOrNullInputsAreNotNull(input, func, analyzer)) {
			return make_uniq<BoundConstantExpression>(Value::BOOLEAN(value.value()));
		}

		return expr;
	}

	if (expr->GetExpressionType() != ExpressionType::OPERATOR_NOT) {
		return expr;
	}

	// The rewrites below are shape-only transformations: NOT(constant_or_null(true, x))
	// becomes constant_or_null(false, x) with x still evaluated per row, and NOT over a
	// boolean constant is pure constant folding. Both keep every expression evaluation
	// intact, so they are safe even in plans that carry side effects (e.g. DML).
	auto &not_expr = expr->Cast<BoundOperatorExpression>();
	D_ASSERT(not_expr.GetChildren().size() == 1);

	auto value = GetBooleanConstant(*not_expr.GetChildren()[0]);
	if (value.has_value()) {
		return make_uniq<BoundConstantExpression>(Value::BOOLEAN(!value.value()));
	}

	value = GetConstantOrNullBoolean(*not_expr.GetChildren()[0]);
	if (!value.has_value()) {
		return expr;
	}

	auto &func = not_expr.GetChildren()[0]->Cast<BoundFunctionExpression>();
	auto &func_children = func.GetChildrenMutable();
	D_ASSERT(func_children.size() >= 2);

	vector<unique_ptr<Expression>> children;
	children.reserve(func_children.size());
	children.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(!value.value())));
	for (idx_t child_idx = 1; child_idx < func_children.size(); ++child_idx) {
		children.push_back(std::move(func_children[child_idx]));
	}

	return ExpressionRewriter::ConstantOrNull(std::move(children), Value::BOOLEAN(!value.value()));
}

unique_ptr<LogicalOperator> ConstantOrNullSimplification::OptimizeFilter(unique_ptr<LogicalOperator> op,
                                                                         bool plan_has_side_effects) {
	auto &filter = op->Cast<LogicalFilter>();
	if (filter.children.size() != 1) {
		return op;
	}

	// Folding constant_or_null into a constant drops the evaluation of the folded-away
	// inputs, so it is only safe when (1) the plan carries no side effects - a DML
	// operator can invalidate the table statistics that NotNullExpressionAnalyzer relies
	// on (e.g. a DML CTE inserts NULLs that the statement's statistics snapshot does not
	// see), and (2) the filter holds no volatile expressions. The NOT(constant_or_null(...))
	// shape rewrite above is unaffected by both checks.
	const bool allow_folding = !plan_has_side_effects && !filter.HasVolatileExpressions();

	NotNullExpressionAnalyzer analyzer(context);
	vector<unique_ptr<Expression>> remaining_expressions;
	remaining_expressions.reserve(filter.expressions.size());
	for (auto &expr : filter.expressions) {
		expr = SimplifyExpression(*filter.children[0], std::move(expr), analyzer, allow_folding);
		auto value = GetBooleanConstant(*expr);
		if (!value.has_value()) {
			remaining_expressions.push_back(std::move(expr));
		} else if (!value.value()) {
			return make_uniq<LogicalEmptyResult>(std::move(op));
		}
	}

	if (!remaining_expressions.empty()) {
		filter.expressions = std::move(remaining_expressions);
		return op;
	}

	if (filter.projection_map.empty()) {
		return std::move(filter.children[0]);
	}

	remaining_expressions.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
	filter.expressions = std::move(remaining_expressions);
	return op;
}

unique_ptr<LogicalOperator> ConstantOrNullSimplification::Optimize(unique_ptr<LogicalOperator> op) {
	// whether the plan carries side effects is computed once at the root: DML operators
	// inside the plan (including DML CTEs) invalidate the statistics-based nullability
	// analysis, so folding must be disabled for the whole plan
	const bool has_side_effects = op->HasSideEffects();
	return OptimizeInternal(std::move(op), has_side_effects);
}

unique_ptr<LogicalOperator> ConstantOrNullSimplification::OptimizeInternal(unique_ptr<LogicalOperator> op,
                                                                           bool plan_has_side_effects) {
	for (auto &child : op->children) {
		child = OptimizeInternal(std::move(child), plan_has_side_effects);
	}

	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		return OptimizeFilter(std::move(op), plan_has_side_effects);
	}

	return op;
}

} // namespace duckdb
