const fs = require('fs');

const JSONToSection = (j) => {
    const header = `### ${j.title}\r\n${j.formatting[0]}\r\n${j.formatting[1]}\r\n${j.formatting[2]}\r\n`
    const body = j.items.reduce((str, item) => {
        str += `|[${item.resource}](${item.link})|${item.progress}|\r\n`
        return str
    }, '')
    return header+body
}

const updateReadMe = (json) => {
    const content = json.map(obj => JSONToSection(obj)).join("\r\n")
    const updatedContent =`# Machine Learning Studies and Adjencent Technologies

## Legend
- ✅ Complete
- 🔄 In Progress
- ⬜ Not Started
- ❌ Postponed

## Run Locally (Do not update ReadMe.md below this line. update using the following commands)
- Have node 20.x or higher
- Create your own repository
- npm install
- npm run update (this will update this readMe)
- npm run push (this will push your changes)\r\n
`.concat(content)
    fs.writeFile('../readMe.md', updatedContent, (err) => {
        if (err) {
            console.error('An error occurred while writing to the file:', err);
            return;
        }
        console.log('Content has been written to the file successfully.');
    });
};

exports.updateReadMe = updateReadMe
