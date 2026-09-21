document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const preview = document.getElementById("preview");
    const previewContainer = document.getElementById("previewContainer");
    const fileName = document.getElementById("fileName");
    const analyzeButton = document.getElementById("analyzeButton");
    const form = document.getElementById("predictionForm");

    imageInput.addEventListener("change", function () {
        const file = this.files[0];
        if (!file) {
            previewContainer.style.display = "none";
            analyzeButton.disabled = true;
            return;
        }

        const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
        if (!allowedTypes.includes(file.type)) {alert("Please select a JPG, JPEG, PNG, or WEBP image.");
            this.value = "";
            previewContainer.style.display = "none";
            analyzeButton.disabled = true;
            return;
        }

        if (file.size > 10 * 1024 * 1024) {alert("The image is too large. Maximum file size is 10 MB.");
            this.value = "";
            previewContainer.style.display = "none";
            analyzeButton.disabled = true;
            return;
        }

        const reader = new FileReader();
        reader.onload = function (event) {preview.src = event.target.result; previewContainer.style.display = "block";};
        reader.readAsDataURL(file);
        fileName.textContent = file.name;
        analyzeButton.disabled = false;
    });
    
    form.addEventListener("submit", function () {
        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing image...";
    });
});