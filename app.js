const logo = document.querySelectorAll('#logo path');

console.log(logo);

let i = 0;
logo.forEach( node => {
    console.log(`Letter ${i} is ${node.getTotalLength()}`);
    i++;
})