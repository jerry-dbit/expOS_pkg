window.onload = () => {
    if(!sessionStorage.name){
        document.getElementById('getStarted').innerText = `Hello!`;
    }
    else{
        document.getElementById('getStarted').innerText = `Hello, ${sessionStorage.name}!`;
    }
} 