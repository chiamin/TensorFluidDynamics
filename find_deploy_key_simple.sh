#!/bin/bash

# Simplified script to find deploy key usage (no jq required)
# Usage: ./find_deploy_key_simple.sh [public_key_file]

echo "=== 查找 Deploy Key 使用情况 ==="
echo ""

# Get the public key to search for
if [ -n "$1" ]; then
    KEY_FILE="$1"
else
    KEY_FILE="~/.ssh/repo1_key.pub"
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "错误: 文件不存在: $KEY_FILE"
    exit 1
fi

# Extract the key content (the actual key part)
KEY_CONTENT=$(cat "$KEY_FILE" | awk '{print $2}' | tr -d '\n')

if [ -z "$KEY_CONTENT" ]; then
    echo "错误: 无法读取 key 内容"
    exit 1
fi

echo "检查 key: $KEY_FILE"
echo "Key 指纹: $(ssh-keygen -lf "$KEY_FILE" 2>/dev/null | awk '{print $2}')"
echo ""

# Get token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "需要 GitHub Personal Access Token"
    echo "创建 token: https://github.com/settings/tokens"
    echo "需要权限: repo (Full control of private repositories)"
    echo ""
    read -p "输入你的 GitHub Personal Access Token: " token
    if [ -z "$token" ]; then
        echo "错误: 需要 token"
        exit 1
    fi
else
    token="$GITHUB_TOKEN"
fi

echo ""
echo "获取所有仓库列表..."

# Get all repositories
page=1
all_repos=""
while true; do
    repos_page=$(curl -s -H "Authorization: token $token" \
        "https://api.github.com/user/repos?per_page=100&page=$page&type=all")
    
    if [ $? -ne 0 ]; then
        echo "错误: 无法获取仓库列表"
        exit 1
    fi
    
    # Check for error message
    if echo "$repos_page" | grep -q '"message"'; then
        echo "错误: $(echo "$repos_page" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)"
        exit 1
    fi
    
    # Extract repo names using grep and sed (no jq needed)
    repo_names=$(echo "$repos_page" | grep -o '"full_name":"[^"]*"' | sed 's/"full_name":"//g' | sed 's/"//g')
    
    if [ -z "$repo_names" ]; then
        break
    fi
    
    all_repos="$all_repos"$'\n'"$repo_names"
    count=$(echo "$repo_names" | wc -l)
    if [ "$count" -lt 100 ]; then
        break
    fi
    page=$((page + 1))
done

repos=$(echo "$all_repos" | grep -v '^$')
total=$(echo "$repos" | wc -l)

if [ "$total" -eq 0 ]; then
    echo "未找到仓库"
    exit 1
fi

echo "找到 $total 个仓库，正在检查..."
echo ""

found=0
count=0
for repo in $repos; do
    count=$((count + 1))
    echo -ne "\r检查进度: $count/$total - $repo                    "
    
    # Get deploy keys for this repo
    deploy_keys=$(curl -s -H "Authorization: token $token" \
        "https://api.github.com/repos/$repo/keys" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$deploy_keys" ]; then
        # Check if our key matches any deploy key (simple grep)
        if echo "$deploy_keys" | grep -q "$KEY_CONTENT"; then
            echo -e "\r✓ 找到: $repo                                    "
            # Extract key title using grep/sed
            key_title=$(echo "$deploy_keys" | grep -o '"title":"[^"]*"' | head -1 | sed 's/"title":"//g' | sed 's/"//g')
            key_id=$(echo "$deploy_keys" | grep -o '"id":[0-9]*' | head -1 | sed 's/"id"://g')
            read_only=$(echo "$deploy_keys" | grep -o '"read_only":[^,}]*' | head -1 | sed 's/"read_only"://g')
            
            if [ -n "$key_title" ]; then
                echo "  Title: $key_title"
            fi
            if [ -n "$key_id" ]; then
                echo "  ID: $key_id"
            fi
            if [ -n "$read_only" ]; then
                echo "  Read-only: $read_only"
            fi
            echo "  链接: https://github.com/$repo/settings/keys"
            echo ""
            found=1
        fi
    fi
done

echo -ne "\r检查完成                                                      "
echo ""
echo ""

if [ $found -eq 0 ]; then
    echo "未找到使用此 key 作为 deploy key 的仓库"
else
    echo "找到 $found 个仓库使用了此 key"
    echo ""
    echo "下一步："
    echo "1. 访问找到的仓库的 Settings → Deploy keys"
    echo "2. 删除该 deploy key"
    echo "3. 然后就可以在个人设置中添加这个 key 了"
fi
