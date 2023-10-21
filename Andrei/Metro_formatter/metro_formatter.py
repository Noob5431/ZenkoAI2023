import json

format = {
    "number" : "safsa",
    "start" : "aswt",
    "end" : "safgas",
    "departures" : {
        "friday" : "ceva",
        "saturday" : "asga",
        "sunday" : "safsa"
    },
    "schedules" : [
        {
            "day" : "thursday",
            "details" : "dasgaes"
        },
        {
            "day" : "monday",
            "details" : "dsafsa"
        }
    ]
}

f = open("metro.json","r")
o = open("metro.txt","w")
x = json.load(f)

output = ""
for i in x:
    output += "Line number: " + str(i["number"]) + "\n"
    output += "Stations are: " + i["start"] + "," + i["end"] + "\n"
    output += "On friday departs from: " + i["departures"]["friday"] + "\n"
    output += "On saturday departs from: " + i["departures"]["saturday"] + "\n"
    output += "On sunday departs from: " + i["departures"]["sunday"] + "\n"
    for j in i["schedules"]:
        output+= "On " + j["day"] + " the schedule is: " + j["details"] + "\n"

o.write(output)