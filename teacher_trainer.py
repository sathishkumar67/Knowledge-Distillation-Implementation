from modules import * 

@dataclass
class Args:
    epochs: int = 15
    lr: float = 6e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = 128
    device_count: int = 1


# instantize the args
args = Args()

# wrapper teacher class
class TeacherMixupModel(L.LightningModule):
    def __init__(self, teacher_model):
        super().__init__()

        self.teacher_model = teacher_model

        # adding mixup data augmentation
        self.mixup = v2.MixUp(alpha=1.0, num_classes=args.num_class)

    def training_step(self, batch, batch_idx):
        loss = 0
        self.teacher_model.train()
        optimizer = self.optimizers()
        optimizer.train()
        optimizer.zero_grad()
        
        batch, label = batch
        # adding mixup data augmentation
        batch, label = self.mixup(batch, label)

        # forward
        teacher_output = self.teacher_model(batch)

        # loss
        loss = F.cross_entropy(teacher_output, label)
        self.log("Train_Loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.teacher_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        self.teacher_model.eval()
        optimizer = self.optimizers()
        optimizer.eval()
        
        batch, label = batch
        # adding mixup data augmentation
        batch, label = self.mixup(batch, label)
        out = self.teacher_model(batch)
        loss = F.cross_entropy(out, label)
        self.log("Val_Loss", loss, prog_bar=True)
        return loss

def main():
    # seed
    torch.manual_seed(args.seed)

    print("Training on", args.device)
    print("Preparing dataset...")
    # prepare dataset
    train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)

    print("Preparing dataloader...")
    # prepare dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Preparing model...")
    # prepare the teacher model
    teacher_model = resnet152(pretrained=False)
    teacher_model.fc = nn.Linear(2048, args.num_class)

    teacher_wrapper_model = TeacherMixupModel(teacher_model=teacher_model)

    # training
    trainer = Trainer(max_epochs=args.epochs,
                  accelerator="cuda")
    trainer.fit(teacher_wrapper_model, train_dataloader, test_dataloader)

    # save the model
    torch.save(teacher_wrapper_model.teacher_model.state_dict(), "teacher.pt")



if __name__ == "__main__":
    main()