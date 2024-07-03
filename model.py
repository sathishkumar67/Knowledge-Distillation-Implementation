from modules import * 
from args import Args

# instantize the args
args = Args()

# base trainer model
class Model(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # storing loss
        self.train_loss = []
        self.val_loss = []
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        self.train_loss.append(loss)
        self.log("Train_Loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        self.val_loss.append(loss)
        self.log("Val_Loss", loss, prog_bar=True)
        return loss
    

# distillation model
class Distillation(L.LightningModule):
    def __init__(self, student_model, teacher_model):
        super().__init__()

        # student and teacher model
        self.student_model = student_model
        self.teacher_model = teacher_model

        # adding mixup data augmentation
        self.mixup = v2.MixUp(alpha=1.0, num_classes=args.num_class)

        # storing loss
        self.train_loss = []
        self.val_loss = []

    def training_step(self, batch, batch_idx):
        loss = 0
        self.student_model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        # adding mixup data augmentation
        batch, label = self.mixup(batch, label)

        # forward
        student_output = self.student_model(batch)
        
        with torch.inference_mode():
            teacher_output = self.teacher_model(batch)

        loss += F.kl_div(F.log_softmax(student_output / args.temperature, dim=1), F.softmax(teacher_output / args.temperature, dim=1), reduction='sum') * (args.temperature**2) / batch.shape[0]
        self.train_loss.append(loss)
        self.log("Train_KL_Loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        self.student_model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        # adding mixup data augmentation
        batch, label = self.mixup(batch, label)
        out = self.student_model(batch)
        loss = F.cross_entropy(out, label)
        self.val_loss.append(loss)
        self.log("Val_Loss", loss, prog_bar=True)
        return loss
