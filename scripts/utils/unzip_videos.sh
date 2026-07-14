#!/bin/bash
# Unzip all LVBench video chunks and extract to videos folder

set -e  # Exit on error

# Configuration
CHUNKS_DIR="${CHUNKS_DIR:-/nas/mars/dataset/LVBench/video_chunks}"
OUTPUT_DIR="${OUTPUT_DIR:-/nas/mars/dataset/LVBench/videos}"
SUBTITLE_DIR="${SUBTITLE_DIR:-/nas/mars/dataset/LVBench/subtitles}"
# Regex for dataset-specific chunk naming (override if needed)
VIDEO_REGEX="${VIDEO_REGEX:-^(video_chunk|videos_chunk|videos_chunked).*[.]zip$}"
# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "============================================"
echo "LVBench Video Extractor"
echo "============================================"
echo ""
echo "Source directory: $CHUNKS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if chunks directory exists
if [ ! -d "$CHUNKS_DIR" ]; then
    echo -e "${RED}Error: Chunks directory not found: $CHUNKS_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build matched video zip list from regex (supports multiple dataset conventions)
VIDEO_ZIPS=()
while IFS= read -r zip_file; do
    filename=$(basename "$zip_file")
    if [[ "$filename" =~ $VIDEO_REGEX ]]; then
        VIDEO_ZIPS+=("$zip_file")
    fi
done < <(find "$CHUNKS_DIR" -maxdepth 1 -type f -name "*.zip" | sort)

TOTAL_ZIPS=${#VIDEO_ZIPS[@]}

if [ "$TOTAL_ZIPS" -eq 0 ]; then
    echo -e "${RED}Error: No video chunk zip files matched regex: $VIDEO_REGEX${NC}"
    exit 1
fi

echo -e "${GREEN}Found $TOTAL_ZIPS video chunk zip files to extract${NC}"
echo "Matching regex: $VIDEO_REGEX"
echo ""

# Extract each zip file
CURRENT=0
FAILED=()

for zip_file in "${VIDEO_ZIPS[@]}"; do
    CURRENT=$((CURRENT + 1))
    filename=$(basename "$zip_file")
    
    echo "============================================"
    echo -e "${YELLOW}[$CURRENT/$TOTAL_ZIPS] Extracting: $filename${NC}"
    echo "============================================"
    
    # Extract with progress
    if unzip -o "$zip_file" -d "$OUTPUT_DIR"; then
        echo -e "${GREEN}✓ Successfully extracted $filename${NC}"
    else
        echo -e "${RED}✗ Failed to extract $filename${NC}"
        FAILED+=("$filename")
    fi
    
    echo ""
done

# Summary
echo "============================================"
echo "Extraction Summary"
echo "============================================"
echo "Total chunks: $TOTAL_ZIPS"
echo "Successfully extracted: $((TOTAL_ZIPS - ${#FAILED[@]}))"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo -e "${GREEN}All video chunks extracted successfully!${NC}"
else
    echo -e "${RED}Failed chunks:${NC}"
    for failed in "${FAILED[@]}"; do
        echo "  - $failed"
    done
fi

if [ -f "$CHUNKS_DIR/subtitle.zip" ] or [ -f "$CHUNKS_DIR/subtitles.zip" ]; then
    mkdir -p "$SUBTITLE_DIR"
    if unzip -o "$CHUNKS_DIR/subtitle.zip" -d "$SUBTITLE_DIR"; then
        echo -e "${GREEN}✓ Successfully extracted subtitles.zip${NC}"
    elif unzip -o "$CHUNKS_DIR/subtitles.zip" -d "$SUBTITLE_DIR"; then
        echo -e "${GREEN}✓ Successfully extracted subtitles.zip${NC}"
    else
        echo -e "${RED}✗ Failed to extract subtitles.zip${NC}"
    fi

else
    echo -e "${RED}✗ No subtitles.zip found in $CHUNKS_DIR${NC}"
fi

echo ""
echo "Output location: $OUTPUT_DIR"

# Count extracted videos
VIDEO_COUNT=$(find "$OUTPUT_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | wc -l)
echo "Total videos extracted: $VIDEO_COUNT"

echo ""
echo "Next steps:"
echo "  1. Verify videos are extracted correctly:"
echo "     ls -lh $OUTPUT_DIR | head -20"
echo ""
echo "  2. Process LVBench dataset:"
echo "     bash scripts/preprocess_lvbench.sh"

