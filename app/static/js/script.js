const gradient = document.querySelector('.gradient');
const uploadVideo = document.querySelector('.uploadVideo');
const title = document.querySelector("h1");
const description = document.querySelector(".description");
const returnToMain = document.querySelector(".returnToMain");
const slider = document.querySelector(".slider");
const fileInput = document.querySelector('#fileInput');
const modal = document.querySelector('.modal');
const logo = document.querySelector('.logo');
const outPut = document.querySelector('.result');


fileInput.addEventListener('change', () => {
  if(fileInput.files.length){
    modal.style.display = "flex";
    description.style.bottom = "-100%";
    uploadVideo.style.bottom = "-100%";
    sendVideo(fileInput.files[0]);
  }
})

uploadVideo.addEventListener('click', () => {
  fileInput.click();
});

returnToMain.addEventListener('click', () => {
  logo.removeAttribute('style');
  outPut.removeAttribute('style');
  title.removeAttribute('style');
  description.removeAttribute('style');
  uploadVideo.removeAttribute('style');
  returnToMain.removeAttribute('style');
  slider.removeAttribute('style');
  gradient.style.backgroundPosition = "center";
})
async function sendVideo(file) {
  fileInput.value = '';
  if (!file) return;
  
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/video', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Network response was not ok');

    const data = await response.json();
   
    const trashType = outPut.querySelector("div");
    trashType.innerHTML = "В кузове самосвала обнаружен " + data.class.toUpperCase();
    const prob = outPut.querySelector("span");
    prob.innerHTML = "Вероятность: " + (data.prod.toFixed(2) * 100) + "%";

    showResult();
    console.log('File uploaded successfully:', data);
  } catch (error) {
    console.error('There was a problem with the fetch operation:', error);
    setTimeout(() => {
      showResult();
    }, 5000);
  }

}

function showResult(){
  outPut.style.left = "10%";
  logo.style.left = "70%";
  gradient.style.backgroundPosition = "0 200%"; 
  title.style.top = "-100%";
  title.style.left = "-100%";
  slider.style.right = "-100%";
  slider.style.bottom = "-100%";
  modal.style.display = "none";
  returnToMain.style.bottom = "20%";
}



let owl = $(".owl-carousel");
owl.owlCarousel({
    items:5,
    loop:true,
    margin: 1,
    autoplay:true,
    autoWidth: false,
    autoplayTimeout:1000,
    autoplayHoverPause:false
});

let typed = new Typed('#typed', {
  stringsElement: '#typed-strings',
  typeSpeed: 16, // Скорость печати
  startDelay: 10, 
  showCursor: false,
  loop: true,
  backSpeed: 20,
  backDelay: 1000,
});
