
const input = document.getElementById('upload');
const infoArea = document.getElementById('upload-label');
const resultDiv = document.getElementById('resultDiv');
const mainAlert = document.getElementById('mainAlert');
const imageResult = document.getElementById('imageResult');
const SelectCaptionType = document.getElementById('SelectCaptionType');
const resultCaption = document.getElementById('resultCaption');
var currentRes = ''
input.addEventListener("change", function (event) {
    if (input.files[0].type != "image/jpeg") {
        alert("Please upload a jpeg image");
        return;
    }
    readURL(input);
    showFileName(event);
    resultDiv.innerHTML = "";
    mainAlert.classList.remove("alert-primary");
    mainAlert.classList.add("alert-warning");
    mainAlert.innerText = "Please wait...";
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("sample_image", input.files[0]);
    xhr.open("POST", "/analyzeImage");
    xhr.send(formData);
    xhr.addEventListener("readystatechange", function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            let jres = JSON.parse(xhr.responseText);
            jres = jres['result']
            currentRes = jres[0][0]
            console.log(currentRes);
            jres.forEach(function (element) {
                const div = document.createElement("div");
                div.className = "alert alert-primary text-center";
                div.setAttribute("role", "alert");
                div.innerHTML = "Prediction (index, name): " + element[0] + ", Score: " + element[1];
                resultDiv.appendChild(div);
            });
            mainAlert.classList.remove("alert-warning");
            mainAlert.classList.add("alert-success");
            mainAlert.innerText = "Here is the analysis result!";
        }
    });
});

SelectCaptionType.addEventListener("change", function () {
    console.log(SelectCaptionType.value);
    let xhr = new XMLHttpRequest();
    xhr.open("GET", "/getCaption?result="+currentRes+"&captionType="+SelectCaptionType.value,true);
    // xhr.send("captionType"+SelectCaptionType.value+"&result"+currentRes);
    xhr.send();
    xhr.addEventListener("readystatechange", function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            let caption = xhr.responseText;
            resultCaption.innerText = caption;
        }else{
            resultCaption.innerText = "Please Wait a moment...";
        }
    });
});

function showFileName(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    console.log(input.files[0].type);
    infoArea.textContent = 'File name: ' + fileName;
}
function readURL(input) {
    if (input.files[0].type != "image/jpeg") {
        alert("Please upload a jpeg image");
        return;
    }
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            imageResult.src = e.target.result;
        };
        reader.readAsDataURL(input.files[0]);
    }
}