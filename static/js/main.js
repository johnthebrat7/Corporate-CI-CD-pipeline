document.addEventListener('DOMContentLoaded', function() {

    // Card animation (smoother)
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100 + (index * 120));
    });

    // Confidence animation
    const confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        const width = confidenceFill.style.width;
        confidenceFill.style.width = '0%';

        setTimeout(() => {
            confidenceFill.style.width = width;
        }, 400);
    }

    // Mouse parallax (smoother)
    document.addEventListener('mousemove', function(e) {
        const shapes = document.querySelectorAll('.shape');
        const x = (e.clientX / window.innerWidth - 0.5) * 20;
        const y = (e.clientY / window.innerHeight - 0.5) * 20;

        shapes.forEach((shape, index) => {
            const depth = (index + 1) * 0.5;
            shape.style.transform = `translate(${x * depth}px, ${y * depth}px)`;
        });
    });

    // Auto dismiss alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'all 0.4s ease';
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => alert.remove(), 400);
        }, 4500);
    });

    // Textarea focus effect
    const textarea = document.querySelector('textarea');
    if (textarea) {
        textarea.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });

        textarea.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    }

});