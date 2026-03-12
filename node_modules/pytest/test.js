#!/usr/bin/env node
//数据传递
// var obj = require('./1.js');
// console.log(obj.a(1,2));
//
// //文件创建
// var fs = require("fs");
// fs.writeFile("tese.txt","hello","utf8",function(){
//     console.log(1);
// });
//
// //第三方模块
// var request = require("request");
// request("http://www.sxuek.com/",function (error,response,body) {
//     console.log(error);
// })

//路径
// var path = require("path");
// console.log(path.resolve("./1.js"));
// console.log(path.dirname("./1.js"));
// console.log(path.join('/aa',"bbb","ccc"));
// console.log(path.basename("/aa/dee/dd/ee.html",".css"))

// console.log(module);

//缓冲器 Buffer将JS的数据处理能力从字符串扩展到了任意二进制数据  v8堆内存
// var buffer = new Buffer([ 0x68, 0x65, 0x6c, 0x6c, 0x6f ]) ;
// var str = buffer.toString("utf-8")
// console.log(str);
//
// var buffer1 = new Buffer("hello","utf-8");
// console.log(buffer1);
//
//  截取后的数组，变值会影响远buffer
// var buffer2 =  new Buffer([ 0x68, 0x65, 0x6c, 0x6c, 0x6f ]) ;
// var sub = buffer2.slice(2);
// console.log(sub);
// sub[0] = 0x65;
// console.log(sub,buffer2);

//复制 拷贝buffer
// var buffer3 = new Buffer([0x68,0x65,0x6c,0x6c,0x6f]);
// var dup = new Buffer(buffer3.length);
// buffer3.copy(dup);
// dup[0]=0x48;
// console.log(buffer3);
// console.log(dup);

//三次握手 四次挥手

var http = require("http");
var path = require("path");
var fs = require("fs");
var config = require("./config.json");
var zlib = require("zlib");
http.createServer(function (req, res) {
    var url = req.url;
    if (req.headers.cookie !== "name=zhangsan" && req.headers.cookie == "undefined") {
        res.setHeader("content-type","text/html;charset=utf-8");
        res.end("login")
    } else {
        if (url == "/favicon.ico") {
            fs.readFile("." + url, function (err, data) {
                if(err){

                }else{
                    res.setHeader("content-type", "application/x-ico");
                    res.end(data);
                }
            })
        } else {
            var root = path.resolve(config.root);
            fs.readdir(root, function (err, data) {
                if (err) {
                    res.writeHead(404, {"content-type": "text/html;charset=utf-8"});
                    res.end("根目录不存在")
                } else {
                    if (path.extname(url)) {
                        var fullUrl = path.join(__dirname, config.root, "." + url);
                    } else {
                        var fullUrl = path.join(__dirname, config.root, "." + url + "/" + config.index);
                    }
                    ;
                    fs.stat(fullUrl, function (err, data) {
                        if (err) {
                            res.writeHead(404, {"content-type": "text/html;charset=utf-8"});
                            res.end("页面不存在")
                        } else {
                            var time1 = new Date(req.headers['if-modified-since']).getTime();
                            var time2 = new Date(data.mtime.toUTCString()).getTime();
                            console.log(time1,time2);
                            if(time1 && time1==time2){
                                res.writeHead(304);
                                res.end();
                            }else{
                                var extname = path.extname(fullUrl);
                                const raw = fs.createReadStream(fullUrl);
                                res.setHeader("Set-Cookie", "name=zhangsan");
                                res.setHeader("Content-Encoding","gzip");
                                res.setHeader("Last-Modified",data.mtime.toUTCString());
                                res.writeHead(200, {"content-type": config.type[extname] + ";charset=utf-8" });
                                raw.pipe(zlib.createGzip()).pipe(res);
                                // res.end(data);
                            }
                        }
                    })
                }
            })
        }
    }
}).listen(config["port"],function () {
    console.log(config["message"]);
});


// server.on("request",function(req,res){
//     console.log(req.url);
//     if(req.url == "/"){
//         res.end("nothing")
//     }else{
//         if(req.url == "/aa"){
//             res.write("123")
//             res.end()
//         }else{
//             res.end("other");
//         }
//     }
// });
