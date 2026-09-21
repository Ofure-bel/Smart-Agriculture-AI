document.addEventListener("DOMContentLoaded", function () {
    const progressBars = document.querySelectorAll(".progress-bar");
    progressBars.forEach(function (bar) {
        const width = parseFloat(bar.dataset.width);
        if (!isNaN(width)) {
            requestAnimationFrame(function () {bar.style.width = width + "%";});
        }
    });
});