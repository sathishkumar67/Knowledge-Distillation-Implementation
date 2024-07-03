from modules import *
from model import *
import argparse as ap

def main(arguments):

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
    # prepare the student and teacher model
    student_model = resnet50(pretrained=False)
    student_model.fc = nn.Linear(2048, args.num_class)

    teacher_model = resnet50(pretrained=True)
    teacher_model.fc = nn.Linear(2048, args.num_class)
    
    # load the teacher model
    if os.path.exists(arguments.model_path):
        teacher_base = resnet152(pretrained=False)
        teacher_base.fc = nn.Linear(2048, 128)
        model = Model.load_from_checkpoint(arguments.model_path,
                                           model=teacher_base)
        teacher_model.load_state_dict(model.model.state_dict())
        print("Teacher model weights have been loaded")
    else:
        raise ValueError("No teacher model found")

    print("Preparing distillation model...")
    # prepare the distillation model
    distill_model = Distillation(student_model, teacher_model)

    print("Start training...")
    # prepare the trainer
    trainer = Trainer(max_epochs=args.epochs, accelerator=args.device, devices=args.device_count)

    # start training
    trainer.fit(distill_model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--if_load_model", type=str, default="yes", help="load model or not for training")
    parser.add_argument("--model_path", type=str, help="model path")
    arguments = parser.parse_args()

    main(arguments)
