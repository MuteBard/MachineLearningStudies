const fs = require('fs');
const simpleGit = require("simple-git")();
const branch = ['main']
const filePaths = ["./readMe.md"];

const getArgs = () => {
    try {
        const data = fs.readFileSync('./src/args.json', 'utf8');
        return data
    } catch (error) {
        console.error('An error occurred while reading the file:', error);
    }
};

const commitPush = async (message, time) => {
    try {
        await addMultipleFiles(filePaths);
        const {message, time} = JSON.parse(getArgs())
        const updatedMessage = buildCommitMessage(message, time);
        await simpleGit.commit(updatedMessage);
        console.log("Changes committed.");

        // Push to the remote repository
        await simpleGit.push("origin", branch);
        console.log(`Changes pushed to remote ${branch}.`);
    } catch (error) {
        console.error("An error occurred:", error);
    }
};

const addMultipleFiles = async (filePaths) => {
    try {
        for (const filePath of filePaths) {
            await simpleGit.add(filePath);
            console.log(`File ${filePath} added.`);
        }
    } catch (error) {
        console.error("An error occurred:", error);
    }
};

const getTruncatedMessage = (commitMessage, limit) => {
    if (commitMessage.length <= limit) {
        return commitMessage;
    } else {
        // Find the last space within the limit
        const truncatedMessage = commitMessage
            .substring(0, limit)
            .replace(/\s\w*$/, "");

        return `${truncatedMessage}...`;
    }
};

const buildCommitMessage = (time, message) => {
    const TEXT_LIMIT = 50;
    const duration = time;
    const updatedMessage = getTruncatedMessage(message, TEXT_LIMIT);
    return `${updatedMessage} - ${duration}`;
};


(async() => {
    await commitPush()
})()