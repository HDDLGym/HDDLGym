document.addEventListener('DOMContentLoaded', function () {
    const toggleButton = document.getElementById('toggle-button');
    const allAccordions = document.querySelectorAll('.accordion-collapse');
    const allAccordionButtons = document.querySelectorAll('.accordion-button');

    let allOpen = true;
    allAccordions.forEach(function (accordion, index) {
        accordion.classList.add('show');
        allAccordionButtons[index].classList.remove('collapsed');
    });

    toggleButton.addEventListener('click', function () {
        allAccordions.forEach(function (accordion, index) {
            if (allOpen) {
                accordion.classList.remove('show');
                allAccordionButtons[index].classList.add('collapsed');
            } else {
                accordion.classList.add('show');
                allAccordionButtons[index].classList.remove('collapsed');
            }
        });
        allOpen = !allOpen;
    });
});