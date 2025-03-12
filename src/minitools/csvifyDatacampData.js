const fs = require('fs-extra');
const path = require("path");
const fileName = process.argv[2];

async function readFile(name) {
    return new Promise((resolve, reject) => {
        fs.readFile(name, 'utf8', (err, data) => {
            if (err) {
                console.log(`File ${name} does not exist`);
                reject(err);
            }
            else {
                console.log(`Accessed ${name}`);
                resolve(data);
            }
        })
    })
}

function createCSVFromString(fileName, content){
    fs.writeFile(fileName, content, (err) => {
        if (err) {
            console.error('Error creating the file:', err);
        } else {
            console.log('File created');
        }
    });    
}

function csvify(rawTableStr){
    let updatedStr = (" "+rawTableStr).replace(/(?![\r\n])\s+/g, ",");
    const csvList = updatedStr.split('\n');
    const updatedCsvList = csvList.map((row) => {
        firstCommaIndex = row.indexOf(',')
        const updatedRow = row.substring(firstCommaIndex + 1);
        return updatedRow;
    })
    return updatedCsvList.join('\n')
}

(async () => {
    const inputDir = path.join(__dirname, "pasteFile.txt");
    const outputDir = path.join(__dirname, fileName);
    const data = await readFile(inputDir);
    createCSVFromString(outputDir, csvify(data))
})()


