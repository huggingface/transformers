// These two things need to be updated at each release for the version selector.
// Last stable version
const stableVersion = "v4.8.2"
// Dictionary doc folder to label. The last stable version should have an empty key.
const versionMapping = {
    "master": "master",
    "": "v4.8.0/v4.8.1/v4.8.2 (stable)",
    "v4.7.0": "v4.7.0",
    "v4.6.0": "v4.6.0",
    "v4.5.1": "v4.5.0/v4.5.1",
    "v4.4.2": "v4.4.0/v4.4.1/v4.4.2",
    "v4.3.3": "v4.3.0/v4.3.1/v4.3.2/v4.3.3",
    "v4.2.2": "v4.2.0/v4.2.1/v4.2.2",
    "v4.1.1": "v4.1.0/v4.1.1",
    "v4.0.1": "v4.0.0/v4.0.1",
    "v3.5.1": "v3.5.0/v3.5.1",
    "v3.4.0": "v3.4.0",
    "v3.3.1": "v3.3.0/v3.3.1",
    "v3.2.0": "v3.2.0",
    "v3.1.0": "v3.1.0",
    "v3.0.2": "v3.0.0/v3.0.1/v3.0.2",
    "v2.11.0": "v2.11.0",
    "v2.10.0": "v2.10.0",
    "v2.9.1": "v2.9.0/v2.9.1",
    "v2.8.0": "v2.8.0",
    "v2.7.0": "v2.7.0",
    "v2.6.0": "v2.6.0",
    "v2.5.1": "v2.5.0/v2.5.1",
    "v2.4.0": "v2.4.0/v2.4.1",
    "v2.3.0": "v2.3.0",
    "v2.2.0": "v2.2.0/v2.2.1/v2.2.2",
    "v2.1.1": "v2.1.1",
    "v2.0.0": "v2.0.0",
    "v1.2.0": "v1.2.0",
    "v1.1.0": "v1.1.0",
    "v1.0.0": "v1.0.0"
}
// The page that have a notebook and therefore should have the open in colab badge.
const hasNotebook = [
    "benchmarks",
    "custom_datasets",
    "multilingual",
    "perplexity",
    "preprocessing",
    "quicktour",
    "task_summary",
    "tokenizer_summary",
    "training"
];

function addIcon() {
    const huggingFaceLogo = "https://huggingface.co/landing/assets/transformers-docs/huggingface_logo.svg";
    const image = document.createElement("img");
    image.setAttribute("src", huggingFaceLogo);

    const div = document.createElement("div");
    div.appendChild(image);
    div.style.textAlign = 'center';
    div.style.paddingTop = '30px';
    div.style.backgroundColor = '#6670FF';

    const scrollDiv = document.querySelector(".wy-side-scroll");
    scrollDiv.prepend(div);
}

function addCustomFooter() {
    const customFooter = document.createElement("div");
    const questionOrIssue = document.createElement("div");
    questionOrIssue.innerHTML = "Stuck? Read our <a href='https://huggingface.co/blog'>Blog posts</a> or <a href='https://github.com/huggingface/transformers'>Create an issue</a>";
    customFooter.appendChild(questionOrIssue);
    customFooter.classList.add("footer");

    const social = document.createElement("div");
    social.classList.add("footer__Social");

    const imageDetails = [
        { link: "https://huggingface.co", imageLink: "https://huggingface.co/landing/assets/transformers-docs/website.svg" },
        { link: "https://twitter.com/huggingface", imageLink: "https://huggingface.co/landing/assets/transformers-docs/twitter.svg" },
        { link: "https://github.com/huggingface", imageLink: "https://huggingface.co/landing/assets/transformers-docs/github.svg" },
        { link: "https://www.linkedin.com/company/huggingface/", imageLink: "https://huggingface.co/landing/assets/transformers-docs/linkedin.svg" }
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
    document.querySelector("footer").appendChild(customFooter);
}

function addGithubButton() {
    const div = `
        <div class="github-repo">
            <a 
                class="github-button"
                href="https://github.com/huggingface/transformers" data-size="large" data-show-count="true" aria-label="Star huggingface/pytorch-transformers on GitHub">
                Star
            </a>
        </div>
    `;
    document.querySelector(".wy-side-nav-search .icon-home").insertAdjacentHTML('afterend', div);
}

function addColabLink() {
    const parts = location.toString().split('/');
    const pageName = parts[parts.length - 1].split(".")[0];

    if (hasNotebook.includes(pageName)) {
        const baseURL = "https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/"
        const linksColab = `
        <div class="colab-dropdown">
            <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
            <div class="colab-dropdown-content">
                <button onclick=" window.open('${baseURL}${pageName}.ipynb')">Mixed</button>
                <button onclick=" window.open('${baseURL}pytorch/${pageName}.ipynb')">PyTorch</button>
                <button onclick=" window.open('${baseURL}tensorflow/${pageName}.ipynb')">TensorFlow</button>
            </div>
        </div>`
        const leftMenu = document.querySelector(".wy-breadcrumbs-aside")
        leftMenu.innerHTML = linksColab + '\n' + leftMenu.innerHTML
    }
}

function addVersionControl() {
    // To grab the version currently in view, we parse the url
    const parts = location.toString().split('/');
    let versionIndex = parts.length - 2;
    // Index page may not have a last part with filename.html so we need to go up
    if (parts[parts.length - 1] != "" && ! parts[parts.length - 1].match(/\.html/)) {
        versionIndex = parts.length - 1;
    }
    // Main classes and models are nested so we need to go deeper
    else if (parts[versionIndex] == "main_classes" || parts[versionIndex] == "model_doc" || parts[versionIndex] == "internal") {
        versionIndex = versionIndex - 1;
    } 
    const version = parts[versionIndex];

    // Menu with all the links,
    const versionMenu = document.createElement("div");

    const htmlLines = [];
    for (const [key, value] of Object.entries(versionMapping)) {
        let baseUrlIndex = (version == "transformers") ? versionIndex + 1: versionIndex;
        var urlParts = parts.slice(0, baseUrlIndex);
        if (key != "") {
            urlParts = urlParts.concat([key]);
        }
        urlParts = urlParts.concat(parts.slice(versionIndex+1));
        htmlLines.push(`<a href="${urlParts.join('/')}">${value}</a>`);
    }

    versionMenu.classList.add("version-dropdown");
    versionMenu.innerHTML = htmlLines.join('\n');
    
    // Button for version selection
    const versionButton = document.createElement("div");
    versionButton.classList.add("version-button");
    let label = (version == "transformers") ? stableVersion : version
    versionButton.innerText = label.concat(" ▼");

    // Toggle the menu when we click on the button
    versionButton.addEventListener("click", () => {
        versionMenu.classList.toggle("version-show");
    });

    // Hide the menu when we click elsewhere
    window.addEventListener("click", (event) => {
        if (event.target != versionButton){
            versionMenu.classList.remove('version-show');
        }
    });

    // Container
    const div = document.createElement("div");
    div.appendChild(versionButton);
    div.appendChild(versionMenu);
    div.style.paddingTop = '25px';
    div.style.backgroundColor = '#6670FF';
    div.style.display = 'block';
    div.style.textAlign = 'center';

    const scrollDiv = document.querySelector(".wy-side-scroll");
    scrollDiv.insertBefore(div, scrollDiv.children[1]);
}

function addHfMenu() {
    const div = `
    <div class="menu">
        <a href="/welcome">🔥 Sign in</a>
        <a href="/models">🚀 Models</a>
        <a href="http://discuss.huggingface.co">💬 Forum</a>
    </div>
    `;
    document.body.insertAdjacentHTML('afterbegin', div);
}

function platformToggle() {
    const codeBlocks = Array.from(document.getElementsByClassName("highlight"));
    const pytorchIdentifier = "## PYTORCH CODE";
    const tensorflowIdentifier = "## TENSORFLOW CODE";

    const promptSpanIdentifier = `<span class="gp">&gt;&gt;&gt; </span>`
    const pytorchSpanIdentifier = `<span class="c1">${pytorchIdentifier}</span>`;
    const tensorflowSpanIdentifier = `<span class="c1">${tensorflowIdentifier}</span>`;

    const getFrameworkSpans = filteredCodeBlock => {
        const spans = filteredCodeBlock.element.innerHTML;
        const pytorchSpanPosition = spans.indexOf(pytorchSpanIdentifier);
        const tensorflowSpanPosition = spans.indexOf(tensorflowSpanIdentifier);

        let pytorchSpans;
        let tensorflowSpans;

        if(pytorchSpanPosition < tensorflowSpanPosition){
            const isPrompt = spans.slice(
                spans.indexOf(tensorflowSpanIdentifier) - promptSpanIdentifier.length,
                spans.indexOf(tensorflowSpanIdentifier)
            ) == promptSpanIdentifier;
            const finalTensorflowSpanPosition = isPrompt ? tensorflowSpanPosition - promptSpanIdentifier.length : tensorflowSpanPosition;

            pytorchSpans = spans.slice(pytorchSpanPosition + pytorchSpanIdentifier.length + 1, finalTensorflowSpanPosition);
            tensorflowSpans = spans.slice(tensorflowSpanPosition + tensorflowSpanIdentifier.length + 1, spans.length);
        }else{
            const isPrompt = spans.slice(
                spans.indexOf(pytorchSpanIdentifier) - promptSpanIdentifier.length,
                spans.indexOf(pytorchSpanIdentifier)
            ) == promptSpanIdentifier;
            const finalPytorchSpanPosition = isPrompt ? pytorchSpanPosition - promptSpanIdentifier.length : pytorchSpanPosition;

            tensorflowSpans = spans.slice(tensorflowSpanPosition + tensorflowSpanIdentifier.length + 1, finalPytorchSpanPosition);
            pytorchSpans = spans.slice(pytorchSpanPosition + pytorchSpanIdentifier.length + 1, spans.length);
        }

        return {
            ...filteredCodeBlock,
            pytorchSample: pytorchSpans ,
            tensorflowSample: tensorflowSpans
        }
    };

    const createFrameworkButtons = sample => {
            const pytorchButton = document.createElement("button");
            pytorchButton.classList.add('pytorch-button')
            pytorchButton.innerText = "PyTorch";

            const tensorflowButton = document.createElement("button");
            tensorflowButton.classList.add('tensorflow-button')
            tensorflowButton.innerText = "TensorFlow";

            const selectorDiv = document.createElement("div");
            selectorDiv.classList.add("framework-selector");
            selectorDiv.appendChild(pytorchButton);
            selectorDiv.appendChild(tensorflowButton);
            sample.element.parentElement.prepend(selectorDiv);

            // Init on PyTorch
            sample.element.innerHTML = sample.pytorchSample;
            pytorchButton.classList.add("selected");
            tensorflowButton.classList.remove("selected");

            pytorchButton.addEventListener("click", () => {
                for(const codeBlock of updatedCodeBlocks){
                    codeBlock.element.innerHTML = codeBlock.pytorchSample;
                }
                Array.from(document.getElementsByClassName('pytorch-button')).forEach(button => {
                    button.classList.add("selected");
                })
                Array.from(document.getElementsByClassName('tensorflow-button')).forEach(button => {
                    button.classList.remove("selected");
                })
            });
            tensorflowButton.addEventListener("click", () => {
                for(const codeBlock of updatedCodeBlocks){
                    codeBlock.element.innerHTML = codeBlock.tensorflowSample;
                }
                Array.from(document.getElementsByClassName('tensorflow-button')).forEach(button => {
                    button.classList.add("selected");
                })
                Array.from(document.getElementsByClassName('pytorch-button')).forEach(button => {
                    button.classList.remove("selected");
                })
            });
        };

    const updatedCodeBlocks = codeBlocks
        .map(element => {return {element: element.firstChild, innerText: element.innerText}})
        .filter(codeBlock => codeBlock.innerText.includes(pytorchIdentifier) && codeBlock.innerText.includes(tensorflowIdentifier))
        .map(getFrameworkSpans)

    updatedCodeBlocks
        .forEach(createFrameworkButtons)
}


/*!
 * github-buttons v2.2.10
 * (c) 2019 なつき
 * @license BSD-2-Clause
 */
/**
 * modified to run programmatically
 */
function parseGithubButtons (){"use strict";var e=window.document,t=e.location,o=window.encodeURIComponent,r=window.decodeURIComponent,n=window.Math,a=window.HTMLElement,i=window.XMLHttpRequest,l="https://unpkg.com/github-buttons@2.2.10/dist/buttons.html",c=i&&i.prototype&&"withCredentials"in i.prototype,d=c&&a&&a.prototype.attachShadow&&!a.prototype.attachShadow.prototype,s=function(e,t,o){e.addEventListener?e.addEventListener(t,o):e.attachEvent("on"+t,o)},u=function(e,t,o){e.removeEventListener?e.removeEventListener(t,o):e.detachEvent("on"+t,o)},h=function(e,t,o){var r=function(n){return u(e,t,r),o(n)};s(e,t,r)},f=function(e,t,o){var r=function(n){if(t.test(e.readyState))return u(e,"readystatechange",r),o(n)};s(e,"readystatechange",r)},p=function(e){return function(t,o,r){var n=e.createElement(t);if(o)for(var a in o){var i=o[a];null!=i&&(null!=n[a]?n[a]=i:n.setAttribute(a,i))}if(r)for(var l=0,c=r.length;l<c;l++){var d=r[l];n.appendChild("string"==typeof d?e.createTextNode(d):d)}return n}},g=p(e),b=function(e){var t;return function(){t||(t=1,e.apply(this,arguments))}},m="body{margin:0}a{color:#24292e;text-decoration:none;outline:0}.octicon{display:inline-block;vertical-align:text-top;fill:currentColor}.widget{ display:inline-block;overflow:hidden;font-family:-apple-system, BlinkMacSystemFont, \"Segoe UI\", Helvetica, Arial, sans-serif;font-size:0;white-space:nowrap;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none}.btn,.social-count{display:inline-block;height:14px;padding:2px 5px;font-size:11px;font-weight:600;line-height:14px;vertical-align:bottom;cursor:pointer;border:1px solid #c5c9cc;border-radius:0.25em}.btn{background-color:#eff3f6;background-image:-webkit-linear-gradient(top, #fafbfc, #eff3f6 90%);background-image:-moz-linear-gradient(top, #fafbfc, #eff3f6 90%);background-image:linear-gradient(180deg, #fafbfc, #eff3f6 90%);background-position:-1px -1px;background-repeat:repeat-x;background-size:110% 110%;border-color:rgba(27,31,35,0.2);-ms-filter:\"progid:DXImageTransform.Microsoft.Gradient(startColorstr='#FFFAFBFC', endColorstr='#FFEEF2F5')\";*filter:progid:DXImageTransform.Microsoft.Gradient(startColorstr='#FFFAFBFC', endColorstr='#FFEEF2F5')}.btn:active{background-color:#e9ecef;background-image:none;border-color:#a5a9ac;border-color:rgba(27,31,35,0.35);box-shadow:inset 0 0.15em 0.3em rgba(27,31,35,0.15)}.btn:focus,.btn:hover{background-color:#e6ebf1;background-image:-webkit-linear-gradient(top, #f0f3f6, #e6ebf1 90%);background-image:-moz-linear-gradient(top, #f0f3f6, #e6ebf1 90%);background-image:linear-gradient(180deg, #f0f3f6, #e6ebf1 90%);border-color:#a5a9ac;border-color:rgba(27,31,35,0.35);-ms-filter:\"progid:DXImageTransform.Microsoft.Gradient(startColorstr='#FFF0F3F6', endColorstr='#FFE5EAF0')\";*filter:progid:DXImageTransform.Microsoft.Gradient(startColorstr='#FFF0F3F6', endColorstr='#FFE5EAF0')}.social-count{position:relative;margin-left:5px;background-color:#fff}.social-count:focus,.social-count:hover{color:#0366d6}.social-count b,.social-count i{position:absolute;top:50%;left:0;display:block;width:0;height:0;margin:-4px 0 0 -4px;border:solid transparent;border-width:4px 4px 4px 0;_line-height:0;_border-top-color:red !important;_border-bottom-color:red !important;_border-left-color:red !important;_filter:chroma(color=red)}.social-count b{border-right-color:#c5c9cc}.social-count i{margin-left:-3px;border-right-color:#fff}.lg .btn,.lg .social-count{height:16px;padding:5px 10px;font-size:12px;line-height:16px}.lg .social-count{margin-left:6px}.lg .social-count b,.lg .social-count i{margin:-5px 0 0 -5px;border-width:5px 5px 5px 0}.lg .social-count i{margin-left:-4px}\n",v={"mark-github":{width:16,height:16,path:'<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>'},eye:{width:16,height:16,path:'<path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/>'},star:{width:14,height:16,path:'<path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/>'},"repo-forked":{width:10,height:16,path:'<path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/>'},"issue-opened":{width:14,height:16,path:'<path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/>'},"cloud-download":{width:16,height:16,path:'<path fill-rule="evenodd" d="M9 12h2l-3 3-3-3h2V7h2v5zm3-8c0-.44-.91-3-4.5-3C5.08 1 3 2.92 3 5 1.02 5 0 6.52 0 8c0 1.53 1 3 3 3h3V9.7H3C1.38 9.7 1.3 8.28 1.3 8c0-.17.05-1.7 1.7-1.7h1.3V5c0-1.39 1.56-2.7 3.2-2.7 2.55 0 3.13 1.55 3.2 1.8v1.2H12c.81 0 2.7.22 2.7 2.2 0 2.09-2.25 2.2-2.7 2.2h-2V11h2c2.08 0 4-1.16 4-3.5C16 5.06 14.08 4 12 4z"/>'}},w={},x=function(e,t,o){var r=p(e.ownerDocument),n=e.appendChild(r("style",{type:"text/css"}));n.styleSheet?n.styleSheet.cssText=m:n.appendChild(e.ownerDocument.createTextNode(m));var a,l,d=r("a",{className:"btn",href:t.href,target:"_blank",innerHTML:(a=t["data-icon"],l=/^large$/i.test(t["data-size"])?16:14,a=(""+a).toLowerCase().replace(/^octicon-/,""),{}.hasOwnProperty.call(v,a)||(a="mark-github"),'<svg version="1.1" width="'+l*v[a].width/v[a].height+'" height="'+l+'" viewBox="0 0 '+v[a].width+" "+v[a].height+'" class="octicon octicon-'+a+'" aria-hidden="true">'+v[a].path+"</svg>"),"aria-label":t["aria-label"]||void 0},[" ",r("span",{},[t["data-text"]||""])]);/\.github\.com$/.test("."+d.hostname)?/^https?:\/\/((gist\.)?github\.com\/[^\/?#]+\/[^\/?#]+\/archive\/|github\.com\/[^\/?#]+\/[^\/?#]+\/releases\/download\/|codeload\.github\.com\/)/.test(d.href)&&(d.target="_top"):(d.href="#",d.target="_self");var u,h,g,x,y=e.appendChild(r("div",{className:"widget"+(/^large$/i.test(t["data-size"])?" lg":"")},[d]));/^(true|1)$/i.test(t["data-show-count"])&&"github.com"===d.hostname&&(u=d.pathname.replace(/^(?!\/)/,"/").match(/^\/([^\/?#]+)(?:\/([^\/?#]+)(?:\/(?:(subscription)|(fork)|(issues)|([^\/?#]+)))?)?(?:[\/?#]|$)/))&&!u[6]?(u[2]?(h="/repos/"+u[1]+"/"+u[2],u[3]?(x="subscribers_count",g="watchers"):u[4]?(x="forks_count",g="network"):u[5]?(x="open_issues_count",g="issues"):(x="stargazers_count",g="stargazers")):(h="/users/"+u[1],g=x="followers"),function(e,t){var o=w[e]||(w[e]=[]);if(!(o.push(t)>1)){var r=b(function(){for(delete w[e];t=o.shift();)t.apply(null,arguments)});if(c){var n=new i;s(n,"abort",r),s(n,"error",r),s(n,"load",function(){var e;try{e=JSON.parse(n.responseText)}catch(e){return void r(e)}r(200!==n.status,e)}),n.open("GET",e),n.send()}else{var a=this||window;a._=function(e){a._=null,r(200!==e.meta.status,e.data)};var l=p(a.document)("script",{async:!0,src:e+(/\?/.test(e)?"&":"?")+"callback=_"}),d=function(){a._&&a._({meta:{}})};s(l,"load",d),s(l,"error",d),l.readyState&&f(l,/de|m/,d),a.document.getElementsByTagName("head")[0].appendChild(l)}}}.call(this,"https://api.github.com"+h,function(e,t){if(!e){var n=t[x];y.appendChild(r("a",{className:"social-count",href:t.html_url+"/"+g,target:"_blank","aria-label":n+" "+x.replace(/_count$/,"").replace("_"," ").slice(0,n<2?-1:void 0)+" on GitHub"},[r("b"),r("i"),r("span",{},[(""+n).replace(/\B(?=(\d{3})+(?!\d))/g,",")])]))}o&&o(y)})):o&&o(y)},y=window.devicePixelRatio||1,C=function(e){return(y>1?n.ceil(n.round(e*y)/y*2)/2:n.ceil(e))||0},F=function(e,t){e.style.width=t[0]+"px",e.style.height=t[1]+"px"},k=function(t,r){if(null!=t&&null!=r)if(t.getAttribute&&(t=function(e){for(var t={href:e.href,title:e.title,"aria-label":e.getAttribute("aria-label")},o=["icon","text","size","show-count"],r=0,n=o.length;r<n;r++){var a="data-"+o[r];t[a]=e.getAttribute(a)}return null==t["data-text"]&&(t["data-text"]=e.textContent||e.innerText),t}(t)),d){var a=g("span",{title:t.title||void 0});x(a.attachShadow({mode:"closed"}),t,function(){r(a)})}else{var i=g("iframe",{src:"javascript:0",title:t.title||void 0,allowtransparency:!0,scrolling:"no",frameBorder:0});F(i,[0,0]),i.style.border="none";var c=function(){var a,d=i.contentWindow;try{a=d.document.body}catch(t){return void e.body.appendChild(i.parentNode.removeChild(i))}u(i,"load",c),x.call(d,a,t,function(e){var a=function(e){var t=e.offsetWidth,o=e.offsetHeight;if(e.getBoundingClientRect){var r=e.getBoundingClientRect();t=n.max(t,C(r.width)),o=n.max(o,C(r.height))}return[t,o]}(e);i.parentNode.removeChild(i),h(i,"load",function(){F(i,a)}),i.src=l+"#"+(i.name=function(e){var t=[];for(var r in e){var n=e[r];null!=n&&t.push(o(r)+"="+o(n))}return t.join("&")}(t)),r(i)})};s(i,"load",c),e.body.appendChild(i)}};t.protocol+"//"+t.host+t.pathname===l?x(e.body,function(e){for(var t={},o=e.split("&"),n=0,a=o.length;n<a;n++){var i=o[n];if(""!==i){var l=i.split("=");t[r(l[0])]=null!=l[1]?r(l.slice(1).join("=")):void 0}}return t}(window.name||t.hash.replace(/^#/,""))):function(t){if(/m/.test(e.readyState)||!/g/.test(e.readyState)&&!e.documentElement.doScroll)setTimeout(t);else if(e.addEventListener){var o=b(t);h(e,"DOMContentLoaded",o),h(window,"load",o)}else f(e,/m/,t)}(function(){for(var t=e.querySelectorAll?e.querySelectorAll("a.github-button"):function(){for(var t=[],o=e.getElementsByTagName("a"),r=0,n=o.length;r<n;r++)~(" "+o[r].className+" ").replace(/[ \t\n\f\r]+/g," ").indexOf(" github-button ")&&t.push(o[r]);return t}(),o=0,r=t.length;o<r;o++)!function(e){k(e,function(t){e.parentNode.replaceChild(t,e)})}(t[o])})};


function onLoad() {
    addIcon();
    addVersionControl();
    addCustomFooter();
    addGithubButton();
    parseGithubButtons();
    addHfMenu();
    addColabLink();
    platformToggle();
}

window.addEventListener("load", onLoad);
