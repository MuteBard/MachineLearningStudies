const fs = require('fs');
const getReadMeText = () => {
    try {
        const data = fs.readFileSync('../readMe.md', 'utf8');
        return data
    } catch (error) {
        console.error('An error occurred while reading the file:', error);
    }
};

const divideIntoList = (text) => {
    const regex = /### (.*?)([\s\S]*?)(?=###|$)/g;
    const sections = [];

    let match;
    while ((match = regex.exec(text)) !== null) {
        sections.push(match[2].trim());
    }
    return sections
}

const sectionToJSON = (section) => {
    const list = section.split('\r\n')
    const title = list[0]

    const items = list.slice(4).map((text) => {
        const regexforSquareBrackets = /\[(.*?)\]/;
        const matchA = text.match(regexforSquareBrackets);
        const resource = matchA[1];

        const regexforParens =/\(([^)]*)\)[^\(\)]*$/;
        const matchB = text.match(regexforParens)
        const link = matchB[1];

        const regexforPipes = /\|([^\|]*)\|\s*([^\|]*)\|/g;
        const matchArray = [...text.matchAll(regexforPipes)];
        const extractedContent = matchArray.map(match => [match[2]]);
        const progress = extractedContent[0][0]

        return {
            resource,
            link,
            progress
        }
    })

    return {
        title,
        formatting: [list[1], list[2], list[3]],
        items
    }
}

function getJson(){
    const text = getReadMeText()
    const sections = divideIntoList(text)
    return sections.map(s => sectionToJSON(s))
}

exports.getJson = getJson;