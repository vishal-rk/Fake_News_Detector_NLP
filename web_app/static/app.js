const newsArticleInput = document.getElementById("news-input");
const predictButton = document.getElementById("predictButton");
const predictionResult = document.getElementById("predictionResult");
const errorSpan = document.getElementById("error-message");

// Event listener for the "Predict" button click
// Event listener for the "Predict" button click
document.addEventListener("DOMContentLoaded", function () {
    // Get the news article text
    const newsArticle = newsArticleInput.value;

    // Clear the error message
    errorSpan.innerHTML = "";

    // Send the news article to the server for prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ newsArticle })
    })
        .then(response => response.json())
        .then(data => {
            console.log("Prediction data:", data); // Debugging log

            // Update the prediction result in the HTML
            const predictionResult = document.getElementById("predictionResult");

            // Wait until the HTML document has finished loading
            predictionResult.addEventListener("DOMContentLoaded", function () {
                if (predictionResult !== null) {
                    predictionResult.innerHTML = `Prediction: ${data.prediction}`;
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            errorSpan.innerHTML = error;
        });
});

