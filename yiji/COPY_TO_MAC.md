# Copy yiji Directory to Mac

## Directory Location
`/raid/guest/OATMeal_Queens/newbie/yiji`

## Method 1: Direct rsync (Recommended - Shows Progress)

**On your Mac terminal:**

```bash
mkdir -p "/Users/sarangchoi/Desktop/y/2025/2025-2/컴퓨터종합설계"

rsync -avz --progress guest111:/raid/guest/OATMeal_Queens/newbie/yiji \
  "/Users/sarangchoi/Desktop/y/2025/2025-2/컴퓨터종합설계/"
```

This will:
- Show progress for each file
- Preserve permissions and timestamps
- Allow resume if interrupted
- Copy the entire `yiji` directory

## Method 2: Create Archive First (Better for Large Directories)

### Step 1: Create Archive on Server

**On the DGX server terminal:**

```bash
cd /raid/guest/OATMeal_Queens/newbie
tar -czf /tmp/yiji.tar.gz yiji
```

### Step 2: Download Archive to Mac

**On your Mac terminal:**

```bash
rsync -avz --progress guest111:/tmp/yiji.tar.gz ~/Desktop/
```

### Step 3: Extract on Mac

**On your Mac terminal:**

```bash
cd "/Users/sarangchoi/Desktop/y/2025/2025-2/컴퓨터종합설계"
tar -xzf ~/Desktop/yiji.tar.gz
```

### Step 4: Clean Up (Optional)

```bash
rm ~/Desktop/yiji.tar.gz
```

## Method 3: Using SCP (Simple but No Progress)

**On your Mac terminal:**

```bash
scp -r guest111:/raid/guest/OATMeal_Queens/newbie/yiji \
  "/Users/sarangchoi/Desktop/y/2025/2025-2/컴퓨터종합설계/"
```

---

## Recommended: Use Method 1 (rsync)

Since you already have `guest111` configured and working, use:

```bash
rsync -avz --progress guest111:/raid/guest/OATMeal_Queens/newbie/yiji \
  "/Users/sarangchoi/Desktop/y/2025/2025-2/컴퓨터종합설계/"
```

This is the easiest and shows progress!

---

## Verify After Copy

**On your Mac:**

```bash
ls -la "/Users/sarangchoi/Desktop/y/2025/2025-2/컴퓨터종합설계/yiji"
```

You should see all the files from the `yiji` directory.

