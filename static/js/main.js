document.addEventListener('DOMContentLoaded', function() {

    // Card animation (gentler fade and smaller travel distance)
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(10px)'; // Reduced from 20px

        setTimeout(() => {
            card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 50 + (index * 80)); // Slightly faster stagger
    });

    // Confidence animation
    const confidenceFill = document.querySelector('.confidence-fill');
    if (confidenceFill) {
        const width = confidenceFill.style.width;
        confidenceFill.style.width = '0%';

        setTimeout(() => {
            confidenceFill.style.width = width;
        }, 300);
    }

    // Mouse parallax (significantly smoothed and reduced intensity)
    document.addEventListener('mousemove', function(e) {
        const shapes = document.querySelectorAll('.shape');
        // Reduced the multiplier from 20 to 5 for a very subtle, mild effect
        const x = (e.clientX / window.innerWidth - 0.5) * 5; 
        const y = (e.clientY / window.innerHeight - 0.5) * 5;

        shapes.forEach((shape, index) => {
            const depth = (index + 1) * 0.3;
            shape.style.transform = `translate(${x * depth}px, ${y * depth}px)`;
            shape.style.transition = 'transform 0.1s linear'; // Added to smooth the mouse tracking
        });
    });

    // Auto dismiss alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'all 0.4s ease';
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-5px)'; // Subtler exit
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
