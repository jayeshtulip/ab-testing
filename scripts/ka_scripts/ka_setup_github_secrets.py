# Ka-MLOps GitHub Secrets Setup Helper
import subprocess
import json

def setup_github_secrets():
    '''Help setup GitHub secrets for Ka-MLOps'''
    
    print(" Ka-MLOps GitHub Secrets Setup")
    print("=" * 50)
    
    print(" Required GitHub Secrets:")
    print("1. AWS_ACCESS_KEY_ID - Your AWS access key")
    print("2. AWS_SECRET_ACCESS_KEY - Your AWS secret key")
    print("")
    
    print(" To add secrets:")
    print("1. Go to your GitHub repository")
    print("2. Click Settings  Secrets and variables  Actions")
    print("3. Click 'New repository secret'")
    print("4. Add each secret with exact names above")
    print("")
    
    # Help get AWS credentials
    print("🔍 Getting your current AWS credentials...")
    try:
        # Try to get current AWS config
        result = subprocess.run(['aws', 'configure', 'list'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(" AWS CLI is configured")
            print(" Current AWS config:")
            print(result.stdout)
        else:
            print(" AWS CLI not configured properly")
    except:
        print(" AWS CLI not found")
    
    print("")
    print(" After adding secrets, your pipeline will:")
    print(" Automatically build and test on every commit")
    print(" Train models and validate performance")
    print(" Deploy to staging and production")
    print(" Run weekly automated retraining")
    
    print("")
    print(" To trigger the pipeline:")
    print("git add .")
    print("git commit -m 'Fix Ka unit tests'")
    print("git push origin main")

if __name__ == "__main__":
    setup_github_secrets()
