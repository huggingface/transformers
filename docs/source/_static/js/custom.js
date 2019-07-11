function addIcon() {
    const huggingFaceLogo = "http://lysand.re/huggingface_logo.svg";
    const image = document.createElement("img");
    image.setAttribute("src", huggingFaceLogo);

    const div = document.createElement("div");
    div.appendChild(image);
    div.style.textAlign = 'center';
    div.style.paddingTop = '30px';
    div.style.backgroundColor = '#6670FF';

    const scrollDiv = document.getElementsByClassName("wy-side-scroll")[0];
    scrollDiv.prepend(div);
}

function addCustomFooter() {
    const customFooter = document.createElement("div");
    const questionOrIssue = document.createElement("div");
    questionOrIssue.innerHTML = "Stuck? Read our <a href='https://medium.com/huggingface'>Blog posts</a> or <a href='https://github.com/huggingface/pytorch_transformers'>Create an issue</a>";
    customFooter.appendChild(questionOrIssue);
    customFooter.classList.add("footer");

    const social = document.createElement("div");
    social.classList.add("footer__Social");

    const imageDetails = [
        { link: "https://huggingface.co", imageLink: "http://lysand.re/icons/website.svg" },
        { link: "https://twitter.com/huggingface", imageLink: "http://lysand.re/icons/twitter.svg" },
        { link: "https://github.com/huggingface", imageLink: "http://lysand.re/icons/github.svg" },
        { link: "https://www.linkedin.com/company/huggingface/", imageLink: "http://lysand.re/icons/linkedin.svg" }
    ];

    imageDetails.forEach(imageLinks => {
        const link = document.createElement("a");
        const image = document.createElement("img");
        image.src = imageLinks.imageLink;
        link.href = imageLinks.link;
        image.style.width = "30px";
        image.classList.add("footer__CustomImage");
        link.appendChild(image);
        social.appendChild(link);
    });

    customFooter.appendChild(social);
    document.getElementsByTagName("footer")[0].appendChild(customFooter);
}

function onLoad() {
    addIcon();
    addCustomFooter();
}

window.addEventListener("load", onLoad);

