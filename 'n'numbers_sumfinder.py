#Progrmam to find the first n natural numbers and the sum of first n natural numbers.
n= int(input("Enter the limit for natural number: "))
m=0
print("The first ",n," natural numbers are: ")
for x in range(1,n+1,1):
  print(x)
  m=m+x
print("The sum is: ", m)
