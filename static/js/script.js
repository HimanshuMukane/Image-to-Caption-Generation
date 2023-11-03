const modeIcon = document.getElementById('mode_icon');
const modeBtn = document.getElementById('mode_btn');
const body = document.querySelector('body');

modeBtn.addEventListener('click', () => {
    body.classList.toggle('bg-dark');
    body.classList.toggle('bg-light');
    modeIcon.classList.toggle('fa-sun');
    modeIcon.classList.toggle('fa-moon');
    if (body.classList.contains('bg-dark')) {
        document.cookie = 'mode=dark;path=/';
    } else {
        document.cookie = 'mode=light;path=/';
    }
});
if(document.cookie.split(';').some((item) => item.trim().startsWith('mode='))) {
    let mode = document.cookie.split(';').find((item) => item.trim().startsWith('mode=')).split('=')[1];
    if (mode == 'dark') {
        body.classList.toggle('bg-dark');
        body.classList.toggle('bg-light');
        modeIcon.classList.toggle('fa-sun');
        modeIcon.classList.toggle('fa-moon');
    }
}