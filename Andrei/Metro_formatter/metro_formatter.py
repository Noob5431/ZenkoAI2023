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

f = open("file.json","w")
x = json.dumps(format,indent=4)
f.write(x)