# Setting up on Windows 11

This gets the whole project running on your PC: the try-on web app and the foot model training. It takes about 30 minutes, most of it downloads. Type every command into **PowerShell** (Start menu, type "PowerShell").

## 1. Install the tools

Run these one at a time. `winget` is Windows' built-in installer.

```powershell
winget install --id Git.Git -e
winget install --id OpenJS.NodeJS.LTS -e
winget install --id astral-sh.uv -e
winget install --id Microsoft.VisualStudioCode -e
```

Close PowerShell and open a new one so it finds the new commands. Then check each one answers with a version number:

```powershell
git --version
node --version
uv --version
nvidia-smi
```

`nvidia-smi` should show your RTX 5000 Ada. If it says "not recognised", install the latest driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) and reboot.

If PowerShell later says "running scripts is disabled on this system", run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**What these are:** Git tracks every change to the code and syncs with GitHub. Node.js runs the web app's tools. uv installs Python and Python packages, each project in its own sandbox (a "virtual environment"), so projects never break each other. VS Code is the editor.

## 2. Get the code

```powershell
mkdir C:\dev
cd C:\dev
git clone https://github.com/aditykulkarni/shoe-tryon.git
cd shoe-tryon
git config user.name "Aditya"
git config user.email "you@example.com"
code .
```

The last line opens the project in VS Code. Install the extensions it suggests, or add **Python** and **Ruff** from the Extensions panel.

## 3. Run the web app

```powershell
npm install
npm test
npm run dev
```

`npm run dev` prints two addresses. On the PC, open the `Local` one. On your phone, connect to the same Wi-Fi and open the `Network` one (like `https://192.168.1.20:5173`). Accept the certificate warning: the dev server uses a home-made HTTPS certificate because phones only allow the camera over HTTPS. If Windows asks whether Node.js may use the network, allow **private networks**.

Every change you save in `src/` shows up on the phone straight away. Press `Ctrl+C` in PowerShell to stop the server.

## 4. Set up training

```powershell
cd training
uv sync
uv run footpose-check
```

`uv sync` downloads Python 3.13 and PyTorch with NVIDIA support, about 3 GB. `footpose-check` should end with your GPU's name and `Ready.`

Then run the tests and a first training run on the built-in toy data (drawn feet, so the whole pipeline can be tried before real photos exist):

```powershell
uv run pytest
uv run footpose-toy-data
uv run footpose-train --data data/toy --epochs 20
```

Each epoch prints a line. Watch `PCK@0.05` climb: it is the share of points that land within 5% of the foot's length from the right spot. Results land in `training/runs/latest/`: `best.pt` is the model, `history.csv` the numbers per epoch (open it in Excel).

Try the model on an image and export it for the browser:

```powershell
uv run footpose-predict runs/latest/best.pt data/toy/val/images/toy_00000.jpg
uv run footpose-export runs/latest/best.pt
```

`uv sync` also wrote `uv.lock`, which pins the exact package versions on your PC. Commit it so the setup is repeatable:

```powershell
git add uv.lock
git commit -m "Lock training dependencies"
git push
```

The first `git push` opens a browser window to sign in to GitHub; that's Git Credential Manager, which came with Git.

## 5. How we work together

- Make a branch for each piece of work: `git switch -c my-change`. Push it and open a pull request on GitHub; I'll review it in our thread.
- Run `uv run ruff check .` and `uv run pytest` (training) or `npm test` (web app) before every commit. GitHub runs them again on every push.
- Stuck on an error? Paste the whole message into our thread. The last lines usually say what went wrong.

Next: [training/README.md](../training/README.md) explains each file and the Python ideas it uses.
