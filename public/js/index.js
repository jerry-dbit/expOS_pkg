window.onload = () => {
    if(!sessionStorage.name){
        location.href = '/login.html';
    }
    else{
        document.getElementsById('greeting').innerText = `Hello, ${sessionStorage.name}!`;
    }
} 