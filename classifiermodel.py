
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from statistics import mean

class bowlingClassifier:
	traininglosses= []
	testinglosses= []
	testaccuracy= []
	totalsteps= []
	a=0

	def _init_(self):
		pass
	
		
	def deviceInit(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("Currently training in : " + self.device.type)

	def modelTrain(self):

		# Base Directory defined.
		base_dir = 'I:\\ML\\Projs\\videoclass\\data'

		# Train and Validation directory Defined.
		train_dir = os.path.join(base_dir, 'train')
		test_dir = os.path.join(base_dir, 'test')

		
		#Defining transforms to feed into DataLoaders
		train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
		                                      transforms.RandomVerticalFlip(),
		                                      transforms.RandomRotation(30),
		                                      transforms.RandomGrayscale(),
		                                      transforms.Resize(255),
		                                      transforms.CenterCrop(224),
		                                      transforms.ToTensor(),
		                                      transforms.Normalize([0.485, 0.456, 0.406],
		                                                           [0.229, 0.224, 0.225])])

		test_transform = transforms.Compose([transforms.Resize(255),
		                                     transforms.CenterCrop(224),
		                                     transforms.ToTensor(),
		                                     transforms.Normalize([0.485, 0.456, 0.406],
		                                                          [0.229, 0.224, 0.225])])


		#arranges the data into suitable structured format for torch to recognise and load in
		train_data = datasets.ImageFolder(train_dir, transform= train_transform)
		test_data = datasets.ImageFolder(test_dir, transform= test_transform)

		#Defining dataloaders with everything attached.
		trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
		testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

		model = models.densenet121(pretrained=True)

		#freeze model parameters
		for params in model.parameters():
		    params.requires_grad = False

		model.classifier = nn.Sequential(nn.Linear(1024, 512),
					                     nn.ReLU(),
					                     nn.Dropout(p=0.2),
					                     nn.Linear(512, 512),
					                     nn.ReLU(),
					                     nn.Dropout(p=0.3),
					                     nn.Linear(512, 2),
					                     nn.LogSoftmax(dim=1))


		#loss criterion and optimiser 
		criterion = nn.NLLLoss()
		optimiser = optim.Adam(model.classifier.parameters(), lr=0.004)

		#move model to GPU or CPU according to their availability.
		model.to(self.device)


		epochs = 1
		steps = 0
		runn = 0
		print_every = 5

		for epoch in range(epochs):
		    for images, labels in trainloader:
		        steps += 1
		        
		        images, labels = images.to(self.device), labels.to(self.device)  #pushing training data onto device
		        
		        optimiser.zero_grad() #erasing preexisting gradients in a classifier
		        
		        logps = model(images) #extracting initial predictions
		        loss = criterion(logps, labels) #calculating initial loss
		        loss.backward() #backpropagates calculated gradients
		        
		        optimiser.step() #one step to update the weights
		        
		        runn += loss.item() #record running loss
		        
		        if steps % print_every == 0:
		            model.eval() #model starts in evaluation mode for testing.
		            test_loss = 0
		            acc = 0
		            
		            for images, labels in testloader: 
		                
		                images, labels= images.to(self.device), labels.to(self.device) #push all testing data
		                
		                logps = model(images) #predictions for testing
		                #loss = Variable(loss, requires_grad = True)
		                loss = criterion(logps, labels) #test loss
		                test_loss += loss.item() #updating test loss
		                
		                ps = torch.exp(logps) 
		                top_ps, top_c = ps.topk(1, dim=1) 
		                equal = top_c == labels.view(top_c.shape) #comparing top_c to actual labels to obtain the number of accurate predictions
		                acc += torch.mean(equal.type(torch.FloatTensor)).item() #storing the result in acc list.
		                
		            self.traininglosses.append(runn/print_every) #running loss/ 5 to average out the loss whenever it gets used
		            self.testinglosses.append(test_loss/len(testloader)) #total loss generated averaged with the test set size.
		            self.testaccuracy.append(acc/len(testloader)) #total accuracy averaged with the test set size
		            self.totalsteps.append(steps) #list containing step count as we proceed
		                
		            print(f"Epoch {epoch+1}/{epochs}.. "
		                  f"Train loss: {runn/print_every:.3f}.. "
		                  f"Test loss: {test_loss/len(testloader):.3f}.. "
		                  f"Test accuracy: {acc/len(testloader):.3f}")
		            runn = 0
		            model.train() #pushes it back to training mode

	def accuracyCalc(self):
		self.a = str(int(mean(self.testaccuracy) * 100))
		print("Test Accuracy:"+self.a+"%")

	def plotShow(self):
		plt.plot(self.totalsteps, self.traininglosses, label='Train Loss')
		plt.plot(self.totalsteps, self.testinglosses, label='Test Loss')
		plt.plot(self.totalsteps, self.testaccuracy, label='Test Accuracy')
		plt.legend()
		plt.grid()
		plt.show()

	def modelSave(self):
		checkpoint = {
		    'parameters' : model.parameters,
		    'state_dict' : model.state_dict()
		}


		torch.save(checkpoint, './bowlingClassifierDensenet121Accuracy{}.pth'.format(a))


	def run(self):
		self.deviceInit()
		self.modelTrain()
		self.accuracyCalc()
		print("Done Training!")


if __name__=='__main__':
    
	# train_bowl_path = 'I:\\ML\\Projs\\videoclass\\data\\train\\bowl'
	# train_rest_path = 'I:\\ML\\Projs\\videoclass\\data\\train\\rest'
	# test_bowl_path = 'I:\\ML\\Projs\\videoclass\\data\\test\\bowl'
	# test_rest_path = 'I:\\ML\\Projs\\videoclass\\data\\test\\rest'

 #    print("Number of training examples of Bowling set:" + (len(os.listdir(train_bowl_path))) + 
 #    	"\nNumber of training examples of Not bowling:" + (len(os.listdir(train_rest_path))) + 
 #    	"\nNumber of testing examples of Bowling set:" + (len(os.listdir(test_bowl_path))) + 
 #    	"\nNumber of testing examples of Not Bowling:" + (len(os.listdir(test_rest_path))) )
 	
    g = bowlingClassifier()
    g.run()

    t1 = input("Want to see a plot of accuracy metrics(Y/N)")

    if(t1 == 'y' or t1 == 'Y'):
    	g.plotShow()

    t2 = 'y' #input("Save the following model?")

    if(t2 == 'y' or t2 == 'Y'):
    	g.modelSave()

