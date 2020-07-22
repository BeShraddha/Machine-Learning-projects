##assigning elements to different lists
l_name = input("Enter different list name").split()
print(l_name)
for i in range(len(l_name)):
    l_name[i] = input("Enter elements for each list").split()
for i in l_name:
    print(i)


##Accessing element from tuple
tup1 = tuple(l_name[0])
print(tup1)
n = int(input("Enter index number"))
print("Element at ",n,"th index is ",tup1[n])


##Deleting element from dictionary
n = int(input("Enter dictionary capacity"))
d = {}
for i in range(n):
    a = input("enter key")
    d[a] = int(input("enter value"))

ele = input("enter key which you want to delete")
try:
    d.pop(ele) 
    print("seccessfuly deleted")
except KeyError:
    print("Key not found") 