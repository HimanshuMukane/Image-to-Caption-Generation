const textinput = document.getElementById('user_input_text');
const choices = document.getElementsByName('userChoice');
const form = document.getElementById('textform');
const resultDiv = document.getElementById('resultDiv');

form.addEventListener("submit", function (e) {
    e.preventDefault();
    submitForm();
})

function submitForm() {
    if (textinput.value == '') {
        alert('Please Enter Some Text');
    }
    else {
        let choice = '';
        let text = textinput.value;
        for (let i = 0; i < choices.length; i++) {
            if (choices[i].checked) {
                choice = choices[i].value;
                break;
            }
        }
        if (choice == '') {
            alert('Please Select A Choice');
        }
        else {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/analyzeText', true);
            xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
            xhr.addEventListener("readystatechange", function () {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    let result = xhr.responseText;
                    resultDiv.innerText = result;
                    console.log(result, 'result');
                } else {
                    resultDiv.innerText = 'Please Wait...';
                }
            })
            xhr.send(`textinput=${text}&userChoice=${choice}`);
        }
    }

}

textinput.addEventListener("keyup", function (e) {
    if (textinput.value == '') {
        resultDiv.innerText = 'Your Result Will Be Shown Here';
    }
})
textinput.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.key == 'Enter') {
        submitForm();
    }
})