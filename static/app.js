// static/app.js

window.addEventListener('scroll', () => {
    // Calculate how far down the page we are scrolled, from 0 to 1
    let scrollPercent = window.scrollY / (document.body.offsetHeight - window.innerHeight);
    
    // Ensure the value stays between 0 and 1
    scrollPercent = Math.min(Math.max(scrollPercent, 0), 1);
    
    // Convert the 0-1 value to a 0%-100% string
    let yPos = scrollPercent * 100;
    
    // Set the CSS variable --scroll-pos-y on the body
    // Our CSS in style.css will use this to move the gradient
    document.body.style.setProperty('--scroll-pos-y', `${yPos}%`);
});