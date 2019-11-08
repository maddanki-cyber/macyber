import csv
import pandas
import math

df = pandas.read_csv('ciphertable.csv')
#print(df)

plain_text1 = int(input("Enter first plaintext value:"))
#print(plain_text1)
cipher_text1 = int(input("Enter first ciphertext value: "))
plain_text2 = input("Enter second plaintext value:")
cipher_text2 = int(input("Enter second ciphertext value:"))


def convert_binrep_int(num):
	digit = str(num)
	binnumber = 0
	multiplier = math.pow(2, (len(digit)-1)) 
	for i in digit:
		binnumber = binnumber + int(i) * multiplier
		multiplier = multiplier/2



	return int(binnumber)


x1 = []
x2 = []
key = []





table = []




#x1
#cipher text
#make entire row a list
x1 = df.iloc[plain_text1].tolist()


#,i all rows for column i, plaintext matches with key
#x2

for i in range(8):
	#print(i)
	z = df.iloc[:,i].tolist()
	#print(z)
	#where is first occurance for cipher text index() in list of column ciphertext and append it to x2
	x2.append(z.index(cipher_text1))

for i in range(len(x2)):
	x2[i] = (x2[i])
	
#print(x2)

#for i in range(8):


#print(x1,x2)
#print(convert_binrep_int(111))


#tuple
#print(convert_int_bin(10001)) 
#checking if match
#i is key in x1
for i in range(8):
	b = convert_binrep_int(x1[i])
	if b in x2:
		#print(x1[i])
		#both keys are below
		#print(i)
		for j in range(8):
			if x2[j] == b:
				#print(j)
				key.append((i,j))


print(key)

convert_p2 = convert_binrep_int(plain_text2)

#convert_c2 = convert_binrep_int(cipher_text2)
print("this is c2" + str(cipher_text2))
for i in range(len(key)):
	a = key[i][0]
	print("This is A" + str(a))
	a2 = key[i][1]
	print("This is a2" + str(a2))
	b = df.iloc[convert_p2,a]
	print("This is b" + str(b))
	d = convert_binrep_int(b)
	c = df.iloc[d,a2]
	print("This is C" + str(c))
	if c == cipher_text2:
		print("Yay you got it: the key is +" + str(key[i]))










	



