#!/bin/bash

# Script to find which repository uses a specific SSH key as deploy key
# Usage: ./find_deploy_key.sh [public_key_file]

echo "=== 查找 Deploy Key 使用情况 ==="
echo ""

# Get the public key to search for
if [ -n "$1" ]; then
    KEY_FILE="$1"
else
    # List available keys
    echo "可用的 SSH keys:"
    keys=($(ls ~/.ssh/*.pub 2>/dev/null))
    if [ ${#keys[@]} -eq 0 ]; then
        echo "未找到 SSH keys"
        exit 1
    fi
    
    for i in "${!keys[@]}"; do
        echo "  [$i] ${keys[$i]}"
    done
    
    echo ""
    read -p "选择要检查的 key 序号: " key_index
    KEY_FILE="${keys[$key_index]}"
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "错误: 文件不存在: $KEY_FILE"
    exit 1
fi

# Extract the key content (the actual key part, not the type or comment)
KEY_CONTENT=$(cat "$KEY_FILE" | awk '{print $2}' | tr -d '\n')

if [ -z "$KEY_CONTENT" ]; then
    echo "错误: 无法读取 key 内容"
    exit 1
fi

echo "检查 key: $KEY_FILE"
echo "Key 指纹: $(ssh-keygen -lf "$KEY_FILE" 2>/dev/null | awk '{print $2}')"
echo ""

# Method 1: Try GitHub CLI first
if command -v gh &> /dev/null; then
    echo "使用 GitHub CLI 检查..."
    
    # Check if logged in
    if ! gh auth status &>/dev/null; then
        echo "GitHub CLI 未登录，请先运行: gh auth login"
        echo ""
        read -p "是否现在登录？(y/n): " login_now
        if [ "$login_now" = "y" ] || [ "$login_now" = "Y" ]; then
            gh auth login
        else
            echo "跳过 GitHub CLI 方法"
        fi
    fi
    
    if gh auth status &>/dev/null; then
        echo "获取所有仓库列表..."
        repos=$(gh repo list --limit 1000 --json nameWithOwner -q '.[].nameWithOwner' 2>/dev/null)
        
        if [ -n "$repos" ]; then
            total=$(echo "$repos" | wc -l)
            echo "找到 $total 个仓库，正在检查..."
            echo ""
            
            found=0
            count=0
            while IFS= read -r repo; do
                count=$((count + 1))
                echo -ne "\r检查进度: $count/$total - $repo                    "
                
                # Get deploy keys for this repo
                deploy_keys=$(gh api "repos/$repo/keys" 2>/dev/null)
                
                if [ $? -eq 0 ] && [ -n "$deploy_keys" ] && [ "$deploy_keys" != "[]" ]; then
                    # Check if our key matches any deploy key
                    if echo "$deploy_keys" | grep -q "$KEY_CONTENT"; then
                        echo -e "\r✓ 找到: $repo                                    "
                        key_info=$(echo "$deploy_keys" | jq -r ".[] | select(.key | contains(\"$KEY_CONTENT\")) | \"  Title: \(.title), ID: \(.id), Read-only: \(.read_only)\"" 2>/dev/null)
                        if [ -n "$key_info" ]; then
                            echo "$key_info"
                        fi
                        echo "  链接: https://github.com/$repo/settings/keys"
                        echo ""
                        found=1
                    fi
                fi
            done <<< "$repos"
            
            echo -ne "\r检查完成                                                      "
            echo ""
            echo ""
            
            if [ $found -eq 0 ]; then
                echo "未找到使用此 key 作为 deploy key 的仓库"
            else
                echo "找到 $found 个仓库使用了此 key"
            fi
            exit 0
        fi
    fi
fi

# Method 2: Use GitHub API with Personal Access Token
echo "使用 GitHub API 方法..."
echo ""

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "错误: 需要安装 jq"
    echo "  Ubuntu/Debian: sudo apt install jq"
    echo "  或安装 GitHub CLI: sudo apt install gh"
    exit 1
fi

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "错误: 需要安装 curl"
    exit 1
fi

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

# Get all repositories (including private)
page=1
all_repos=""
while true; do
    repos_page=$(curl -s -H "Authorization: token $token" \
        "https://api.github.com/user/repos?per_page=100&page=$page&type=all")
    
    if [ $? -ne 0 ] || echo "$repos_page" | jq -e '.message' &>/dev/null; then
        echo "错误: 无法获取仓库列表"
        echo "$repos_page" | jq -r '.message' 2>/dev/null || echo "$repos_page"
        exit 1
    fi
    
    repo_names=$(echo "$repos_page" | jq -r '.[].full_name' 2>/dev/null)
    
    if [ -z "$repo_names" ] || [ "$repo_names" = "" ]; then
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
        # Check if our key matches any deploy key
        if echo "$deploy_keys" | grep -q "$KEY_CONTENT"; then
            echo -e "\r✓ 找到: $repo                                    "
            key_info=$(echo "$deploy_keys" | jq -r ".[] | select(.key | contains(\"$KEY_CONTENT\")) | \"  Title: \(.title), ID: \(.id), Read-only: \(.read_only)\"" 2>/dev/null)
            if [ -n "$key_info" ]; then
                echo "$key_info"
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
    echo ""
    echo "可能的原因："
    echo "1. 这个 key 确实没有被用作 deploy key"
    echo "2. 这个 key 可能在其他账户的仓库中"
    echo "3. 检查所有页面（如果仓库超过 100 个）"
else
    echo "找到 $found 个仓库使用了此 key"
    echo ""
    echo "下一步："
    echo "1. 访问找到的仓库的 Settings → Deploy keys"
    echo "2. 删除该 deploy key"
    echo "3. 然后就可以在个人设置中添加这个 key 了"
fi
