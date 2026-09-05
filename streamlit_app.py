import os
import sqlite3
from datetime import datetime, date
from typing import Optional

import pandas as pd
import streamlit as st

# OpenAI is optional. The app can still run without an API key.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MoneyWise",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_FILE = "moneywise.db"


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_type TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            transaction_date TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL UNIQUE,
            amount REAL NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS savings_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            target REAL NOT NULL,
            saved REAL NOT NULL DEFAULT 0
        )
        """
    )

    conn.commit()
    conn.close()


initialize_database()


# ============================================================
# DATABASE HELPERS
# ============================================================

def add_transaction(
    transaction_type: str,
    amount: float,
    category: str,
    description: str,
    transaction_date: str,
):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO transactions
        (transaction_type, amount, category, description, transaction_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            transaction_type,
            amount,
            category,
            description,
            transaction_date,
        ),
    )

    conn.commit()
    conn.close()


def get_transactions():
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            id,
            transaction_type,
            amount,
            category,
            description,
            transaction_date
        FROM transactions
        ORDER BY transaction_date DESC, id DESC
        """,
        conn,
    )

    conn.close()
    return df


def delete_transaction(transaction_id: int):
    conn = get_connection()

    conn.execute(
        "DELETE FROM transactions WHERE id = ?",
        (transaction_id,),
    )

    conn.commit()
    conn.close()


def get_budgets():
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT id, category, amount
        FROM budgets
        ORDER BY category
        """,
        conn,
    )

    conn.close()
    return df


def save_budget(category: str, amount: float):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO budgets (category, amount)
        VALUES (?, ?)
        ON CONFLICT(category)
        DO UPDATE SET amount = excluded.amount
        """,
        (category, amount),
    )

    conn.commit()
    conn.close()


def delete_budget(budget_id: int):
    conn = get_connection()

    conn.execute(
        "DELETE FROM budgets WHERE id = ?",
        (budget_id,),
    )

    conn.commit()
    conn.close()


def get_savings_goals():
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT id, name, target, saved
        FROM savings_goals
        ORDER BY id DESC
        """,
        conn,
    )

    conn.close()
    return df


def add_savings_goal(name: str, target: float):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO savings_goals (name, target, saved)
        VALUES (?, ?, 0)
        """,
        (name, target),
    )

    conn.commit()
    conn.close()


def update_savings_goal(goal_id: int, saved: float):
    conn = get_connection()

    conn.execute(
        """
        UPDATE savings_goals
        SET saved = ?
        WHERE id = ?
        """,
        (saved, goal_id),
    )

    conn.commit()
    conn.close()


def delete_savings_goal(goal_id: int):
    conn = get_connection()

    conn.execute(
        "DELETE FROM savings_goals WHERE id = ?",
        (goal_id,),
    )

    conn.commit()
    conn.close()


# ============================================================
# FINANCIAL CALCULATIONS
# ============================================================

def get_financial_summary():
    df = get_transactions()

    if df.empty:
        return {
            "income": 0.0,
            "expenses": 0.0,
            "balance": 0.0,
            "transaction_count": 0,
        }

    income = df.loc[
        df["transaction_type"] == "Income",
        "amount",
    ].sum()

    expenses = df.loc[
        df["transaction_type"] == "Expense",
        "amount",
    ].sum()

    return {
        "income": float(income),
        "expenses": float(expenses),
        "balance": float(income - expenses),
        "transaction_count": len(df),
    }


def get_category_spending():
    df = get_transactions()

    if df.empty:
        return pd.DataFrame(columns=["category", "amount"])

    expenses = df[df["transaction_type"] == "Expense"]

    if expenses.empty:
        return pd.DataFrame(columns=["category", "amount"])

    result = (
        expenses.groupby("category")["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )

    return result


# ============================================================
# AI
# ============================================================

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def build_business_context():
    summary = get_financial_summary()
    transactions = get_transactions()
    budgets = get_budgets()
    goals = get_savings_goals()

    recent_transactions = transactions.head(15).to_dict(
        orient="records"
    )

    budget_data = budgets.to_dict(
        orient="records"
    )

    goal_data = goals.to_dict(
        orient="records"
    )

    return {
        "summary": summary,
        "recent_transactions": recent_transactions,
        "budgets": budget_data,
        "savings_goals": goal_data,
    }


def fallback_ai_response(question: str):
    summary = get_financial_summary()
    category_spending = get_category_spending()

    question_lower = question.lower()

    if summary["transaction_count"] == 0:
        return (
            "I don't have enough transaction data yet. "
            "Add some business income and expenses and I can "
            "start analyzing your cash flow."
        )

    if "cash flow" in question_lower or "cashflow" in question_lower:
        if summary["income"] == 0:
            return (
                "Your business has no recorded income yet. "
                "Add your income transactions so I can evaluate "
                "your cash-flow position."
            )

        margin = (
            (summary["income"] - summary["expenses"])
            / summary["income"]
        ) * 100

        return (
            f"Your recorded income is KSh {summary['income']:,.2f} "
            f"and expenses are KSh {summary['expenses']:,.2f}. "
            f"Your current recorded cash position is "
            f"KSh {summary['balance']:,.2f}. "
            f"Your expense-to-income position is approximately "
            f"{100 - margin:.1f}%."
        )

    if (
        "spending" in question_lower
        or "expense" in question_lower
    ):
        if category_spending.empty:
            return "There are no recorded expenses yet."

        top = category_spending.iloc[0]

        return (
            f"Your largest recorded spending category is "
            f"{top['category']} at KSh {top['amount']:,.2f}. "
            "Review this category first when looking for "
            "possible cost savings."
        )

    if "profit" in question_lower:
        profit = summary["income"] - summary["expenses"]

        return (
            f"Based on the transactions recorded, your current "
            f"gross cash surplus is KSh {profit:,.2f}."
        )

    return (
        f"Your business currently has KSh "
        f"{summary['balance']:,.2f} in recorded net cash position. "
        "I can analyze your cash flow, spending, budgets, "
        "pricing decisions, or financial performance."
    )


def ask_ai(question: str):
    client = get_openai_client()
    context = build_business_context()

    if client is None:
        return fallback_ai_response(question)

    system_prompt = """
You are MoneyWise, an AI-powered financial copilot designed
for small African businesses.

Your job is to help business owners make better everyday
financial decisions.

You specialize in:
- cash-flow management
- budgeting
- pricing
- transaction analysis
- expense management
- revenue analysis
- savings planning
- financial decision support

Use the business data supplied by the application.

Important rules:
1. Be practical and easy to understand.
2. Use KSh when discussing Kenyan currency unless the user
   specifies another currency.
3. Never invent financial data.
4. Clearly distinguish recorded data from estimates.
5. Give actionable recommendations.
6. When discussing pricing, consider costs, margin and
   sustainability.
7. When discussing cash flow, focus on money coming in,
   money going out and timing.
8. Do not claim to be a licensed financial adviser.
9. Do not make investment guarantees.
10. Keep responses concise unless more detail is requested.
"""

    user_prompt = f"""
Business financial context:

{context}

Business owner's question:

{question}
"""

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=system_prompt,
            input=user_prompt,
        )

        return response.output_text

    except Exception:
        return fallback_ai_response(question)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }

        .moneywise-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0;
        }

        .moneywise-subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-top: 0;
        }

        .metric-card {
            padding: 20px;
            border-radius: 15px;
            background: white;
            border: 1px solid #e2e8f0;
            margin-bottom: 10px;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.9rem;
        }

        .metric-value {
            font-size: 1.7rem;
            font-weight: 750;
        }

        .insight-card {
            padding: 18px;
            border-radius: 14px;
            background: white;
            border: 1px solid #e2e8f0;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 💰 MoneyWise")
st.sidebar.caption(
    "Your AI-powered financial copilot for business."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Transactions",
        "Cash Flow",
        "Budgets",
        "Pricing",
        "Savings Goals",
        "AI Copilot",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "MoneyWise helps small businesses understand their "
    "money and make better everyday financial decisions."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="moneywise-title">MoneyWise</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="moneywise-subtitle">'
        "AI-powered financial copilot for your business"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    summary = get_financial_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Cash Position",
            f"KSh {summary['balance']:,.2f}",
        )

    with col2:
        st.metric(
            "Total Income",
            f"KSh {summary['income']:,.2f}",
        )

    with col3:
        st.metric(
            "Total Expenses",
            f"KSh {summary['expenses']:,.2f}",
        )

    with col4:
        st.metric(
            "Transactions",
            f"{summary['transaction_count']:,}",
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Cash Flow Overview")

        df = get_transactions()

        if df.empty:
            st.info(
                "No transactions yet. Add your first income "
                "or expense transaction."
            )
        else:
            chart_df = df.copy()

            chart_df["date"] = pd.to_datetime(
                chart_df["transaction_date"]
            )

            chart_df["signed_amount"] = chart_df.apply(
                lambda row: (
                    row["amount"]
                    if row["transaction_type"] == "Income"
                    else -row["amount"]
                ),
                axis=1,
            )

            daily = (
                chart_df.groupby("date")["signed_amount"]
                .sum()
                .sort_index()
            )

            st.line_chart(daily)

    with right:
        st.subheader("Top Spending Categories")

        category_spending = get_category_spending()

        if category_spending.empty:
            st.info("No expenses recorded yet.")
        else:
            st.bar_chart(
                category_spending.set_index("category")
            )

    st.subheader("MoneyWise Insight")

    if summary["transaction_count"] == 0:
        st.info(
            "Start by recording your business transactions. "
            "MoneyWise will use them to generate financial insights."
        )
    else:
        profit = summary["income"] - summary["expenses"]

        if profit > 0:
            st.success(
                f"Your recorded income currently exceeds your "
                f"expenses by KSh {profit:,.2f}."
            )
        elif profit < 0:
            st.warning(
                f"Your recorded expenses currently exceed income "
                f"by KSh {abs(profit):,.2f}."
            )
        else:
            st.info(
                "Your recorded income and expenses are currently equal."
            )


# ============================================================
# TRANSACTIONS
# ============================================================

elif page == "Transactions":

    st.title("💳 Everyday Transactions")
    st.write(
        "Record the money coming into and going out of your business."
    )

    tab1, tab2 = st.tabs(
        ["Add Transaction", "Transaction History"]
    )

    with tab1:

        transaction_type = st.selectbox(
            "Transaction Type",
            ["Income", "Expense"],
        )

        if transaction_type == "Income":
            categories = [
                "Sales",
                "Service Revenue",
                "Other Income",
            ]
        else:
            categories = [
                "Inventory",
                "Transport",
                "Rent",
                "Utilities",
                "Marketing",
                "Salaries",
                "Supplies",
                "Other Expense",
            ]

        category = st.selectbox(
            "Category",
            categories,
        )

        amount = st.number_input(
            "Amount (KSh)",
            min_value=0.0,
            step=100.0,
            format="%.2f",
        )

        description = st.text_input(
            "Description",
            placeholder="e.g. Customer payment for order #102",
        )

        transaction_date = st.date_input(
            "Date",
            value=date.today(),
        )

        if st.button(
            "Save Transaction",
            type="primary",
            use_container_width=True,
        ):

            if amount <= 0:
                st.error("Enter an amount greater than zero.")
            else:
                add_transaction(
                    transaction_type,
                    amount,
                    category,
                    description,
                    transaction_date.isoformat(),
                )

                st.success(
                    f"{transaction_type} transaction saved successfully."
                )

                st.rerun()

    with tab2:

        transactions = get_transactions()

        if transactions.empty:
            st.info("No transactions have been recorded yet.")
        else:

            display_df = transactions.copy()

            display_df["amount"] = display_df["amount"].apply(
                lambda x: f"KSh {x:,.2f}"
            )

            display_df = display_df.rename(
                columns={
                    "id": "ID",
                    "transaction_type": "Type",
                    "amount": "Amount",
                    "category": "Category",
                    "description": "Description",
                    "transaction_date": "Date",
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Delete Transaction")

            transaction_id = st.number_input(
                "Transaction ID",
                min_value=1,
                step=1,
            )

            if st.button("Delete Transaction"):

                delete_transaction(int(transaction_id))

                st.success("Transaction deleted.")

                st.rerun()


# ============================================================
# CASH FLOW
# ============================================================

elif page == "Cash Flow":

    st.title("💰 Cash Flow")

    st.write(
        "Understand how money moves through your business."
    )

    summary = get_financial_summary()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Money In",
            f"KSh {summary['income']:,.2f}",
        )

    with col2:
        st.metric(
            "Money Out",
            f"KSh {summary['expenses']:,.2f}",
        )

    with col3:
        st.metric(
            "Net Cash Flow",
            f"KSh {summary['balance']:,.2f}",
        )

    st.divider()

    transactions = get_transactions()

    if transactions.empty:
        st.info("Add transactions to see your cash flow.")
    else:

        chart_df = transactions.copy()

        chart_df["date"] = pd.to_datetime(
            chart_df["transaction_date"]
        )

        chart_df["income"] = chart_df.apply(
            lambda x: x["amount"]
            if x["transaction_type"] == "Income"
            else 0,
            axis=1,
        )

        chart_df["expense"] = chart_df.apply(
            lambda x: x["amount"]
            if x["transaction_type"] == "Expense"
            else 0,
            axis=1,
        )

        daily = (
            chart_df.groupby("date")[["income", "expense"]]
            .sum()
            .sort_index()
        )

        st.subheader("Income vs Expenses")

        st.line_chart(daily)

        st.subheader("Cash Flow Analysis")

        if summary["income"] > 0:

            expense_ratio = (
                summary["expenses"]
                / summary["income"]
            ) * 100

            st.write(
                f"Your recorded expenses represent "
                f"approximately {expense_ratio:.1f}% "
                f"of your recorded income."
            )

            if expense_ratio > 90:
                st.warning(
                    "Your expenses are very close to your income. "
                    "Consider reviewing major costs and protecting "
                    "your operating cash."
                )

            elif expense_ratio > 70:
                st.info(
                    "A significant portion of recorded income is "
                    "being consumed by expenses. Monitor your "
                    "largest cost categories closely."
                )

            else:
                st.success(
                    "Your recorded expenses are currently below "
                    "70% of recorded income."
                )


# ============================================================
# BUDGETS
# ============================================================

elif page == "Budgets":

    st.title("📊 Business Budgets")

    st.write(
        "Set spending limits and track how your business is performing "
        "against them."
    )

    with st.form("budget_form"):

        category = st.selectbox(
            "Expense Category",
            [
                "Inventory",
                "Transport",
                "Rent",
                "Utilities",
                "Marketing",
                "Salaries",
                "Supplies",
                "Other Expense",
            ],
        )

        amount = st.number_input(
            "Monthly Budget (KSh)",
            min_value=0.0,
            step=500.0,
            format="%.2f",
        )

        submitted = st.form_submit_button(
            "Save Budget",
            use_container_width=True,
        )

        if submitted:

            if amount <= 0:
                st.error("Budget must be greater than zero.")
            else:
                save_budget(category, amount)

                st.success(
                    f"Budget for {category} saved."
                )

                st.rerun()

    st.divider()

    budgets = get_budgets()
    spending = get_category_spending()

    if budgets.empty:
        st.info(
            "No budgets have been created yet."
        )
    else:

        for _, budget in budgets.iterrows():

            category = budget["category"]
            budget_amount = float(budget["amount"])

            spent = 0.0

            if not spending.empty:

                matching = spending[
                    spending["category"] == category
                ]

                if not matching.empty:
                    spent = float(
                        matching.iloc[0]["amount"]
                    )

            remaining = budget_amount - spent

            st.subheader(category)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Budget",
                    f"KSh {budget_amount:,.2f}",
                )

            with col2:
                st.metric(
                    "Spent",
                    f"KSh {spent:,.2f}",
                )

            with col3:
                st.metric(
                    "Remaining",
                    f"KSh {remaining:,.2f}",
                )

            progress = min(
                spent / budget_amount,
                1.0,
            )

            st.progress(progress)

            if spent > budget_amount:
                st.error(
                    f"You are over the {category} budget by "
                    f"KSh {spent - budget_amount:,.2f}."
                )
            elif spent >= budget_amount * 0.8:
                st.warning(
                    f"You have used {progress * 100:.0f}% "
                    f"of this budget."
                )
            else:
                st.success(
                    f"You have used {progress * 100:.0f}% "
                    f"of this budget."
                )

            if st.button(
                f"Delete {category} budget",
                key=f"delete_budget_{budget['id']}",
            ):

                delete_budget(int(budget["id"]))

                st.rerun()


# ============================================================
# PRICING
# ============================================================

elif page == "Pricing":

    st.title("🏷️ Smart Pricing")

    st.write(
        "Estimate a sustainable selling price using your costs "
        "and desired profit margin."
    )

    col1, col2 = st.columns(2)

    with col1:

        product_name = st.text_input(
            "Product or Service",
            placeholder="e.g. Handmade bag",
        )

        unit_cost = st.number_input(
            "Cost per Unit (KSh)",
            min_value=0.0,
            step=10.0,
            format="%.2f",
        )

        other_costs = st.number_input(
            "Other Cost per Unit (KSh)",
            min_value=0.0,
            step=10.0,
            format="%.2f",
        )

    with col2:

        desired_margin = st.number_input(
            "Desired Profit Margin (%)",
            min_value=0.0,
            max_value=95.0,
            value=30.0,
            step=1.0,
        )

        expected_units = st.number_input(
            "Expected Units Sold",
            min_value=1,
            step=1,
        )

    if st.button(
        "Calculate Price",
        type="primary",
        use_container_width=True,
    ):

        total_unit_cost = unit_cost + other_costs

        if total_unit_cost <= 0:
            st.error(
                "Enter at least one cost."
            )
        else:

            selling_price = (
                total_unit_cost
                / (1 - desired_margin / 100)
            )

            profit_per_unit = (
                selling_price - total_unit_cost
            )

            expected_profit = (
                profit_per_unit * expected_units
            )

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Suggested Minimum Price",
                    f"KSh {selling_price:,.2f}",
                )

            with col2:
                st.metric(
                    "Profit / Unit",
                    f"KSh {profit_per_unit:,.2f}",
                )

            with col3:
                st.metric(
                    "Expected Profit",
                    f"KSh {expected_profit:,.2f}",
                )

            if product_name:
                st.success(
                    f"For {product_name}, a price of approximately "
                    f"KSh {selling_price:,.2f} gives you a "
                    f"{desired_margin:.0f}% margin based on the "
                    "costs entered."
                )

            st.info(
                "This is a cost-and-margin estimate. Before setting "
                "a final price, compare it with customer demand, "
                "competitor prices and your business's operating costs."
            )


# ============================================================
# SAVINGS GOALS
# ============================================================

elif page == "Savings Goals":

    st.title("🎯 Business Savings Goals")

    st.write(
        "Create financial goals and track your progress."
    )

    with st.form("goal_form"):

        goal_name = st.text_input(
            "Goal Name",
            placeholder="e.g. New equipment",
        )

        target = st.number_input(
            "Target Amount (KSh)",
            min_value=0.0,
            step=500.0,
            format="%.2f",
        )

        submitted = st.form_submit_button(
            "Create Goal",
            use_container_width=True,
        )

        if submitted:

            if not goal_name.strip():
                st.error("Enter a goal name.")

            elif target <= 0:
                st.error(
                    "Target amount must be greater than zero."
                )

            else:
                add_savings_goal(
                    goal_name.strip(),
                    target,
                )

                st.success("Savings goal created.")

                st.rerun()

    st.divider()

    goals = get_savings_goals()

    if goals.empty:
        st.info(
            "You have not created any savings goals yet."
        )

    else:

        for _, goal in goals.iterrows():

            target = float(goal["target"])
            saved = float(goal["saved"])

            progress = min(
                saved / target,
                1.0,
            )

            st.subheader(goal["name"])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Target",
                    f"KSh {target:,.2f}",
                )

            with col2:
                st.metric(
                    "Saved",
                    f"KSh {saved:,.2f}",
                )

            with col3:
                st.metric(
                    "Remaining",
                    f"KSh {max(target - saved, 0):,.2f}",
                )

            st.progress(progress)

            st.caption(
                f"{progress * 100:.1f}% complete"
            )

            if saved >= target:
                st.success(
                    "🎉 Goal reached!"
                )

            new_saved = st.number_input(
                "Update amount saved (KSh)",
                min_value=0.0,
                value=saved,
                step=100.0,
                key=f"saved_{goal['id']}",
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "Update Goal",
                    key=f"update_{goal['id']}",
                ):

                    update_savings_goal(
                        int(goal["id"]),
                        new_saved,
                    )

                    st.success("Goal updated.")

                    st.rerun()

            with col2:

                if st.button(
                    "Delete Goal",
                    key=f"delete_goal_{goal['id']}",
                ):

                    delete_savings_goal(
                        int(goal["id"])
                    )

                    st.rerun()


# ============================================================
# AI COPILOT
# ============================================================

elif page == "AI Copilot":

    st.title("🤖 MoneyWise AI Copilot")

    st.write(
        "Ask MoneyWise questions about your business finances."
    )

    st.info(
        "Examples: "
        "“How is my cash flow?” · "
        "“Where am I spending the most?” · "
        "“Am I overspending?” · "
        "“How can I improve my margins?”"
    )

    question = st.text_area(
        "Ask MoneyWise",
        placeholder=(
            "Ask a question about your business finances..."
        ),
        height=130,
    )

    if st.button(
        "Ask MoneyWise",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():
            st.warning(
                "Please enter a question."
            )
        else:

            with st.spinner(
                "MoneyWise is analyzing your business..."
            ):

                answer = ask_ai(
                    question.strip()
                )

            st.markdown("### MoneyWise's Insight")

            st.markdown(
                f'<div class="insight-card">{answer}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader("Quick Questions")

    quick_questions = [
        "How is my cash flow?",
        "Where am I spending the most?",
        "Am I spending too much?",
        "What should I watch in my finances?",
        "How can I improve my profit?",
    ]

    for quick_question in quick_questions:

        if st.button(
            quick_question,
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing your business..."
            ):

                answer = ask_ai(
                    quick_question
                )

            st.markdown("### MoneyWise's Insight")

            st.markdown(
                f'<div class="insight-card">{answer}</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    f"MoneyWise • {datetime.now().year}"
)

st.sidebar.caption(
    "Financial insights are based on information recorded "
    "in the application."
)