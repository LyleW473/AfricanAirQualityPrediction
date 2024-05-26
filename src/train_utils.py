import pandas as pd
import os

def create_submission(predictions, test_set):
    if not os.path.exists("submissions"):
        os.makedirs("submissions")
    submission = pd.DataFrame({'id': test_set["id"], 'pm2_5': predictions})

    print(submission)
    print(submission.shape)

    num_submissions = len(os.listdir("submissions"))
    submission.to_csv(f"submissions/submission_{num_submissions}.csv", index=False)
    return submission