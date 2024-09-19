document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('expense-form');
    const categorySelect = document.getElementById('category');
    const expenseList = document.getElementById('expense-list');
    const additionalCategories = [];

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const amount = document.getElementById('amount').value;
        const category = categorySelect.value;
        const description = document.getElementById('description').value;

        const response = await fetch('/add-expense', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ amount, category, description })
        });

        if (response.ok) {
            const expense = await response.json();
            addExpenseToList(expense);
        }
    });

    function addExpenseToList(expense) {
        const listItem = document.createElement('li');
        listItem.textContent = `${expense.amount} - ${expense.category} - ${expense.description}`;
        expenseList.appendChild(listItem);
    }

    function addAdditionalCategory(category) {
        if (additionalCategories.length < 5) {
            additionalCategories.push(category);
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            categorySelect.appendChild(option);
        } else {
            alert('You can only add up to 5 additional categories.');
        }
    }
});
