from typing import Optional
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def guide_mongodb_setup() -> Optional[str]:
    """
    Interactive guide for MongoDB Atlas setup

    Returns:
        Connection string, or None if user cancels
    """

    console.print(Panel.fit(
        "[bold cyan]MongoDB Atlas Setup[/bold cyan]\n\n"
        "We'll help you set up MongoDB Atlas (free tier) for vector search.",
        border_style="cyan"
    ))

    # Step 1: Account
    console.print("\n[bold]Step 1: Create MongoDB Atlas Account[/bold]\n")

    has_account = questionary.confirm(
        "Do you already have a MongoDB Atlas account?",
        default=False
    ).ask()

    if not has_account:
        console.print(Markdown("""
1. Go to: https://www.mongodb.com/cloud/atlas/register
2. Sign up (free, no credit card required)
3. Verify your email
4. Come back here when done
"""))

        questionary.press_any_key_to_continue("Press any key when account is ready...").ask()

    # Step 2: Cluster
    console.print("\n[bold]Step 2: Create a Free Cluster[/bold]\n")

    has_cluster = questionary.confirm(
        "Do you already have a cluster created?",
        default=False
    ).ask()

    if not has_cluster:
        console.print(Markdown("""
1. In MongoDB Atlas dashboard, click "**Build a Database**"
2. Choose **M0 FREE** tier
3. Select a cloud provider (AWS recommended)
4. Choose a region close to you
5. Name your cluster (e.g., "codecontext-cluster")
6. Click "**Create**"
7. Wait 1-3 minutes for cluster to deploy
"""))

        questionary.press_any_key_to_continue("Press any key when cluster is ready...").ask()

    # Step 3: Database User
    console.print("\n[bold]Step 3: Create Database User[/bold]\n")

    console.print(Markdown("""
1. Click "**Database Access**" in left sidebar
2. Click "**Add New Database User**"
3. Choose "**Password**" authentication
4. Username: `codecontext` (or your choice)
5. Click "**Autogenerate Secure Password**" (save it!)
6. Database User Privileges: "**Read and write to any database**"
7. Click "**Add User**"
"""))

    questionary.press_any_key_to_continue("Press any key when user is created...").ask()

    # Step 4: Network Access
    console.print("\n[bold]Step 4: Configure Network Access[/bold]\n")

    console.print(Markdown("""
1. Click "**Network Access**" in left sidebar
2. Click "**Add IP Address**"
3. Click "**Allow Access from Anywhere**" (for development)
   - This adds `0.0.0.0/0` (allow all IPs)
   - For production, restrict to your IP
4. Click "**Confirm**"
"""))

    questionary.press_any_key_to_continue("Press any key when network access is configured...").ask()

    # Step 5: Connection String
    console.print("\n[bold]Step 5: Get Connection String[/bold]\n")

    console.print(Markdown("""
1. Go back to "**Database**" in left sidebar
2. Click "**Connect**" on your cluster
3. Choose "**Connect your application**"
4. Driver: **Python**
5. Version: **3.6 or later**
6. Copy the connection string (looks like):
```
   mongodb+srv://codecontext:<password>@cluster0.<id>.mongodb.net/?retryWrites=true&w=majority
```
7. Replace `<password>` with your actual password
"""))

    connection_string = questionary.text(
        "Paste your connection string here:",
        validate=lambda x: len(x) > 20 and "mongodb" in x
    ).ask()

    if not connection_string:
        console.print("[red]Setup cancelled[/red]")
        return None

    # Step 6: Test Connection
    console.print("\n[bold]Step 6: Test Connection[/bold]\n")

    with console.status("[cyan]Testing connection..."):
        try:
            from pymongo import MongoClient
            import certifi

            client = MongoClient(
                connection_string,
                tlsCAFile=certifi.where()
            )
            # Try to connect
            client.admin.command('ping')
            client.close()

            console.print("[green]✓ Connection successful![/green]")

        except Exception as e:
            console.print(f"[red]✗ Connection failed: {e}[/red]")
            retry = questionary.confirm("Try a different connection string?").ask()
            if retry:
                return guide_mongodb_setup()
            return None

    # Success!
    console.print(Panel.fit(
        "[bold green]✓ MongoDB Atlas Setup Complete![/bold green]\n\n"
        "Connection string will be saved to your config.",
        border_style="green"
    ))

    return connection_string


def show_vector_index_instructions():
    """
    Show instructions for creating vector search index

    This needs to be done AFTER first indexing
    """
    console.print(Panel.fit(
        "[bold cyan]Create Vector Search Index[/bold cyan]\n\n"
        "After your first indexing, you need to create a search index.",
        border_style="cyan"
    ))

    console.print(Markdown("""
## Creating Vector Search Index

**IMPORTANT:** Do this AFTER running `codecontext index` for the first time!

1. Go to your MongoDB Atlas dashboard
2. Click on your cluster → "**Browse Collections**"
3. You should see:
   - Database: `codecontext`
   - Collection: `codebase`
4. Click the "**Search Indexes**" tab (NOT "Indexes")
5. Click "**Create Search Index**"
6. Choose "**JSON Editor**"
7. Paste this configuration:
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

8. Index name: `vector_index`
9. Click "**Create Search Index**"
10. Wait 1-2 minutes for index to build

**You only need to do this ONCE per cluster.**
"""))
