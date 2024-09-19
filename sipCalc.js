document.addEventListener('DOMContentLoaded', function() {
    const incomeSlider = document.getElementById('incomeRange');
    const incomeValue = document.getElementById('incomeValue');
    incomeSlider.addEventListener('input', function() {
        incomeValue.textContent = incomeSlider.value;
    });

    const yearsSlider = document.getElementById('yearsRange');
    const yearsValue = document.getElementById('yearsValue');
    yearsSlider.addEventListener('input', function() {
        yearsValue.textContent = yearsSlider.value;
    });

    const goalSlider = document.getElementById('goalRange');
    const goalValue = document.getElementById('goalValue');
    goalSlider.addEventListener('input', function() {
        goalValue.textContent = goalSlider.value;
    });

    document.getElementById('sipForm').addEventListener('submit', function(event) {
        event.preventDefault();

        let inc = parseFloat(incomeSlider.value);
        let year = parseFloat(yearsSlider.value);
        let FV = parseFloat(goalSlider.value);
        let RP = parseInt(document.getElementById('risk').value);

        let r;
        let n = year * 12;
        let SIP;

        switch (RP) {
            case 1:
                r = 0.12;
                break;
            case 2:
                r = 0.15;
                break;
            case 3:
                r = 0.20;
                break;
            default:
                document.getElementById('result').innerText = "Invalid risk profile entered.";
                return;
        }

        let numerator = FV * (r / 12);
        let denominator = Math.pow(1 + (r / 12), n) - 1;
        SIP = numerator / denominator;

        let resultText = `Your monthly savings should be: ${SIP.toFixed(2)}`;
        
        if (SIP > inc) {
            alert("\nYour goal is too ambitious!");
        }

        document.getElementById('result').innerText = resultText; // Display result on the webpage
    });
});
