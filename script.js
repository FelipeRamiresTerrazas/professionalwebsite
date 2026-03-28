// ========================================
// DYNAMIC YEARS OF EXPERIENCE
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    var words = ['one','two','three','four','five','six','seven','eight','nine','ten'];
    var years = new Date().getFullYear() - 2019;

    // Metric card number (e.g. "7+")
    var metricEl = document.getElementById('years-experience');
    if (metricEl) {
        metricEl.textContent = years + '+';
    }

    // Prose text (e.g. "seven years")
    var textEl = document.getElementById('years-experience-text');
    if (textEl) {
        var word = years >= 1 && years <= words.length ? words[years - 1] : years;
        textEl.textContent = word + ' years';
    }
});

// ========================================
// EXPERTISE MASTER-DETAIL
// ========================================

function switchExpertise(clickedItem, panelId) {
    // Update nav active state
    document.querySelectorAll('.exp-nav-item').forEach(function(item) {
        item.classList.remove('active');
    });
    clickedItem.classList.add('active');

    // Fade out all panels
    var panels = document.querySelectorAll('.exp-panel');
    panels.forEach(function(panel) {
        panel.classList.remove('is-active');
        setTimeout(function() {
            if (!panel.classList.contains('is-active')) {
                panel.classList.remove('is-visible');
            }
        }, 280);
    });

    // Fade in target panel
    var target = document.getElementById('exp-panel-' + panelId);
    if (target) {
        target.classList.add('is-visible');
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                target.classList.add('is-active');
            });
        });
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var firstNavItem = document.querySelector('.exp-nav-item');
    if (firstNavItem) {
        firstNavItem.classList.add('active');
        var panelId = firstNavItem.getAttribute('data-panel');
        var firstPanel = document.getElementById('exp-panel-' + panelId);
        if (firstPanel) {
            firstPanel.classList.add('is-visible');
            // Small delay so the transition fires after display: block
            requestAnimationFrame(function() {
                requestAnimationFrame(function() {
                    firstPanel.classList.add('is-active');
                });
            });
        }
    }
});

// ========================================
// TAB SWITCHING FUNCTIONALITY
// ========================================

function switchTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
    });

    // Show the selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Highlight the selected tab button
    const selectedBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }
}

// Tab button event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Only bind tab switching to buttons that have a data-tab attribute
    const tabButtons = document.querySelectorAll('.tab-btn[data-tab]');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
});

// ========================================
// PORTFOLIO ACCORDION
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.getElementById('portfolioToggle');
    const submenu = document.getElementById('portfolioSubmenu');

    if (toggle && submenu) {
        toggle.addEventListener('click', function(e) {
            // Don't trigger the normal tab-switch logic
            e.stopPropagation();
            const isOpen = submenu.classList.contains('open');
            submenu.classList.toggle('open');
            toggle.setAttribute('aria-expanded', String(!isOpen));
        });
    }
});

// ========================================
// INTERSECTION OBSERVER FOR ANIMATIONS
// ========================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-up');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements for animation on scroll
document.addEventListener('DOMContentLoaded', function() {
    const elements = document.querySelectorAll('.expertise-card, .timeline-item, .skill-category, .about-text');
    elements.forEach(el => observer.observe(el));
});

// ========================================
// FORM HANDLING
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    const contactForm = document.getElementById('contactForm');
    
    if (contactForm) {
        contactForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const endpoint = contactForm.dataset.formspreeEndpoint;
            const submitButton = contactForm.querySelector('button[type="submit"]');
            const defaultButtonText = submitButton ? submitButton.textContent : 'Send Message';
            
            // Simple validation
            const name = this.querySelector('input[type="text"]').value.trim();
            const email = this.querySelector('input[type="email"]').value.trim();
            const message = this.querySelector('textarea').value.trim();
            
            if (!name || !email || !message) {
                showNotification('Please fill in all fields', 'error');
                return;
            }
            
            // Email validation
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                showNotification('Please enter a valid email address', 'error');
                return;
            }

            if (!endpoint || endpoint.includes('YOUR_FORM_ID')) {
                showNotification('Form is not configured yet. Add your Formspree form ID in index.html.', 'error');
                return;
            }
            
            const formData = new FormData(this);

            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = 'Sending...';
            }

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                if (response.ok) {
                    showNotification('Message sent successfully! Thank you for reaching out.', 'success');
                    contactForm.reset();
                } else {
                    showNotification('Could not send message. Please try again in a moment.', 'error');
                }
            } catch (error) {
                showNotification('Network error. Please check your connection and try again.', 'error');
            } finally {
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.textContent = defaultButtonText;
                }
            }
        });
    }
});

// ========================================
// NOTIFICATION SYSTEM
// ========================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add styles dynamically
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background-color: ${type === 'success' ? '#00cc88' : type === 'error' ? '#ff6b6b' : '#0066cc'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 2000;
        animation: slideIn 0.3s ease;
        max-width: 300px;
    `;
    
    document.body.appendChild(notification);
    
    // Remove notification after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// ========================================
// NAVBAR ACTIVE STATE ON SCROLL
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    window.addEventListener('scroll', () => {
        let current = '';
        
        const sections = document.querySelectorAll('section');
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').slice(1) === current) {
                link.classList.add('active');
            }
        });
    });
});

// ========================================
// PARALLAX EFFECT
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    const heroAccent = document.querySelector('.hero-accent');
    
    if (heroAccent) {
        window.addEventListener('scroll', () => {
            const scrollPosition = window.pageYOffset;
            heroAccent.style.transform = `translateY(${scrollPosition * 0.5}px)`;
        });
    }
});

// ========================================
// KEYBOARD NAVIGATION
// ========================================

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const navLinks = document.querySelector('.nav-links');
        if (navLinks && navLinks.style.display === 'flex') {
            navLinks.style.display = 'none';
        }
    }
});

// ========================================
// ADD ANIMATION STYLES
// ========================================

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
    
    .nav-links a.active {
        color: var(--primary-color);
        font-weight: 600;
    }
`;
document.head.appendChild(style);

// ========================================
// SCROLL TO TOP BUTTON
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    // Create scroll to top button
    const scrollButton = document.createElement('button');
    scrollButton.innerHTML = '↑';
    scrollButton.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        padding: 12px 16px;
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 1.5rem;
        cursor: pointer;
        display: none;
        z-index: 999;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    `;
    
    document.body.appendChild(scrollButton);
    
    // Show/hide button based on scroll position
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            scrollButton.style.display = 'flex';
            scrollButton.style.alignItems = 'center';
            scrollButton.style.justifyContent = 'center';
        } else {
            scrollButton.style.display = 'none';
        }
    });
    
    // Scroll to top when clicked
    scrollButton.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    // Hover effect
    scrollButton.addEventListener('mouseover', () => {
        scrollButton.style.transform = 'scale(1.1)';
    });
    
    scrollButton.addEventListener('mouseout', () => {
        scrollButton.style.transform = 'scale(1)';
    });
});

// ========================================
// PERFORMANCE: LAZY LOAD IMAGES
// ========================================

if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src || img.src;
                img.classList.add('loaded');
                imageObserver.unobserve(img);
            }
        });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => imageObserver.observe(img));
}

// ========================================
// MOBILE MENU CLOSE ON OUTSIDE CLICK
// ========================================

document.addEventListener('click', function(e) {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelector('.nav-links');
    const hamburger = document.querySelector('.hamburger');
    
    if (navbar && navLinks && hamburger) {
        if (!navbar.contains(e.target) && navLinks.style.display === 'flex') {
            navLinks.style.display = 'none';
        }
    }
});

// ========================================
// PREVENT CONSOLE ERRORS IN PRODUCTION
// ========================================

if (typeof console === 'object') {
    console.log('%cWelcome to Felipe Ramires Terrazas Portfolio', 'color: #0066cc; font-size: 16px; font-weight: bold;');
}
