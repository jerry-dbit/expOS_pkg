// form loading animation
document.addEventListener("DOMContentLoaded", () => {
    // Your existing code here
    const form = [...document.querySelector('.form').children];
    
    form.forEach((item, i) => {
        setTimeout(() => {
            item.style.opacity = 1;
        }, i*100);
    })
    
    window.onload = () => {
        if(sessionStorage.name){
            location.href = '/';
        }
    }
    
    // form validation
    
    const username = document.querySelector('.name') || null;
    const email = document.querySelector('.email');
    const password = document.querySelector('.password');
    const submitBtn = document.querySelector('.submitBtn');
    
    
    
    if(username == null){ // means login page is open
        submitBtn.addEventListener('click', () => {
            fetch('http://localhost:3000/login-user',{
                method: 'post',
                headers: new Headers({'Content-Type': 'application/json'}),
                body: JSON.stringify({
                    email: email.value,
                    password: password.value
                })
            })
            .then(res => res.json())
            .then(data => {
                validateData(data);
            })
        })
    } else{ // means register page is open
    
        submitBtn.addEventListener('click', () => {
            fetch('http://localhost:3000/register-user', {
                method: 'post',
                headers: new Headers({'Content-Type': 'application/json'}),
                body: JSON.stringify({
                    name: username.value,
                    email: email.value,
                    password: password.value
                })
            })
            .then(res => res.json())
            .then(data => {
                validateData(data);
            })
        })
    
    }
    const validateData = (data) => {
        if(!data.name){
            alertBox(data);
        } else{
            sessionStorage.name = data.name;
            sessionStorage.email = data.email;
            location.href = '/index.html';
        }
    
    }
    
    const alertBox = (data) => {
        const alertContainer = document.querySelector('.alert-box');
        const alertMsg = document.querySelector('.alert');
        alertMsg.innerHTML = data;
    
        alertContainer.style.top = `5%`;
        setTimeout(() => {
            alertContainer.style.top = null;
        }, 5000);
    }
});