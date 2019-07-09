function addIcon() {
    const huggingFaceLogo = "http://lysand.re/huggingface_logo.svg";
    const image = document.createElement("img");
    image.setAttribute("src", huggingFaceLogo)


    const div = document.createElement("div")
    div.appendChild(image);
    div.style.textAlign = 'center';
    div.style.paddingTop = '30px';
    div.style.backgroundColor = '#6670FF'

    const scrollDiv = document.getElementsByClassName("wy-side-scroll")[0];
    scrollDiv.prepend(div)
}

window.addEventListener("load", addIcon)

