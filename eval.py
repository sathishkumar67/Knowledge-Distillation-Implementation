from modules import *
from args import Args
from model import Distillation

# instantize the args
args = Args()

def main(arguments):
    checkpoints_names = ["teacher.pt", "student.pt", "distilled_student.pt"]
    for ckpt_name in checkpoints_names:
        hf_hub_download(repo_id='pt-sk/knowledge_distillation', filename=f'{ckpt_name}', local_dir="/checkpoints")

    print("Downloaded The Checkpoints")

    # trained studeht model
    student_model = resnet50(pretrained=False)
    student_model.fc = nn.Linear(2048, args.num_class)
    # move model to GPU if available
    student_model.to(args.device)
    # load the student model
    student_model.load_state_dict(torch.load("/checkpoints/student.pt"))
    print("Student model weights have been loaded")

    # trained teacher model
    teacher_model = resnet152(pretrained=True)
    teacher_model.fc = nn.Linear(2048, args.num_class)
    # move model to GPU if available
    teacher_model.to(args.device)
    # load the teacher model
    teacher_model.load_state_dict(torch.load("/checkpoints/teacher.pt"))
    print("Teacher model weights have been loaded")

    # trained distilled student model
    distilled_student_model = resnet50(pretrained=False)
    distilled_student_model.fc = nn.Linear(2048, args.num_class)
    # move model to GPU if available
    distilled_student_model.to(args.device)
    # load the distilled student model
    distilled_student_model.load_state_dict(torch.load("/checkpoints/distilled_student.pt"))
    print("Distilled student model weights have been loaded")

    # prepare dataset
    test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)

    # prepare dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # calculate the accuracy, precision, recall and f1 score for all student_model, teacher_model, distilled_student model
    metrics = {}

    for model in [student_model, teacher_model, distilled_student_model]:
        model.eval()
        with torch.no_grad():
            all_labels = []
            all_preds = []
            for batch, labels in test_dataloader:
                outputs = model(batch.to(args.device))
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        metrics[str(model)] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    # convert the metrics dictionary to a pandas dataframe
    df = pd.DataFrame.from_dict(metrics, orient='index')

    # print the dataframe
    print(df)

    # save the dataframe to a csv file
    df.to_csv('distillaion_metrics_results.csv')
     
if __name__ == "__main__":
    main()