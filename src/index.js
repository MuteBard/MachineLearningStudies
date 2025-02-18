const fs = require('fs');
const { getJson } = require("./getJsonFromReadMe");
const { updateReadMe } = require("./updateReadMe");
const { commitPush } = require('./commitPush')

const json = getJson();

window.addEventListener("keydown", function (event) {
    if (event.keyCode === 27 || event.key === "Escape") {
        process.exit(1);
    }
});

const populateUnits = (json) => {
    const unitDropDown = document.getElementById("selectUnit");
    json.forEach((unit) => {
        const option = document.createElement("option");
        option.text = unit.title;
        option.value = unit.title;
        unitDropDown.appendChild(option);
    });
};

const populateTitles = (json) => {
    const unitDropDown = document.getElementById("selectUnit");
    unitDropDown.addEventListener("click", async () => {
        const titleDropDown = document.getElementById("selectTitle");
        titleDropDown.innerHTML = '';
        const obj = json.find((unit) => unit.title === unitDropDown.value);

        obj?.items.forEach((item) => {
            const option = document.createElement("option");
            option.text = item.resource;
            option.value = item.resource;
            titleDropDown.appendChild(option);
        });
    });
};

const populateProgress = () => {
    const progressDropDown = document.getElementById("selectProgress");
    ["⬜", "✅", "🔄", "❌"].forEach((item) => {
        const option = document.createElement("option");
        option.text = item;
        option.value = item;
        progressDropDown.appendChild(option);
    });
};

const populateTime = () => {
    const timeDropDown = document.getElementById("selectTime");
    [
        "Completed",
        "1 Hour Progress",
        "2 Hour Progress",
        "3 Hour Progress",
        "10 Minute Progress",
        "20 Minute Progress",
        "30 Minute Progress"
    ].forEach((item) => {
        const option = document.createElement("option");
        option.text = item;
        option.value = item;
        timeDropDown.appendChild(option);
    });
};


const updateJsonWithInput = (json) => {
    const unitDropDownValue = document.getElementById("selectUnit")?.value;
    const titleDropDownValue = document.getElementById("selectTitle")?.value;
    const progressDropDownValue = document.getElementById("selectProgress")?.value;

    const unitIndex = json?.findIndex((unit) => unit.title === unitDropDownValue);
    const itemIndex = json[unitIndex]?.items.findIndex((item) => item.resource ===  titleDropDownValue);

    json[unitIndex].items[itemIndex].progress = progressDropDownValue
    return json
}

const commitMessageData = (json) => {
    const titleDropDownValue = document.getElementById("selectTitle").value;
    const timeDropDownValue = document.getElementById("selectTime").value;
    return {message:titleDropDownValue, time:timeDropDownValue}
}

const submitMessage = () => {
    const container = document.getElementById("bigContainer");
    container.innerHTML = '';

    const newHeading = document.createElement("h2");
    newHeading.textContent = "Updating readme.md...";
    container.appendChild(newHeading);
}

document.getElementById("commitChanges").addEventListener("click", async () => {
    const updatedJson = updateJsonWithInput(json);
    createArgs()
    updateReadMe(updatedJson)
    submitMessage()
    setTimeout(() => process.exit(1), 3000);
    
});

const createArgs = () => {
    fs.writeFile('args.json', JSON.stringify(commitMessageData(json)), (err) => {
        if (err) {
            console.error('An error occurred while writing to the file:', err);
            return;
        }
        console.log('Content has been written to the file successfully.');
    });
}

populateUnits(json);
populateTitles(json);
populateProgress();
populateTime();