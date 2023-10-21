import json

f = open("bus.txt","r")
o = open("bus_formatted.txt","w")

string = f.read()
index = 0
output = ""
for line in string.splitlines():
    rem = index%6
    if rem == 0:
        output += "The line is: "+ line + "\n"
    if rem == 1:
        output += "In the direction: "+ line + "\n"
    if rem == 2:
        output += "Starting from: " + line
    if rem == 3:
        output += " " + line + "\n"
    if rem == 4:
        output += "Ending on: " + line +"\n"
    if rem == 5:
        output += "Details: " + line + "\n"
    index += 1

o.write(output)