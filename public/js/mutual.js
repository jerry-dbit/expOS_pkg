document.addEventListener('DOMContentLoaded', async () => {
    window.onload = async () => {
        try {
            const symbolsObj = {
                "0P00005UN0.BO": true,
            };
            
            const response = await fetch(`http://localhost:3000/mquotes`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(symbolsObj),
            });

            if (response.ok) {
                const result = await response.json();
                const icici = result[0]?.regularMarketPrice
                if (price) {
                    document.getElementById("mutual").innerText = `Market Price: ${icici}`;
                } else {
                    document.getElementById("mutual").innerText = "No market price available.";
                }
            } else {
                console.error('Failed to fetch: ', response.status);
            }
        } catch (error) {
            console.log(error);
        }
    };
});
