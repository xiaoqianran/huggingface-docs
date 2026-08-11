<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 程序化用户访问控制管理

本指南介绍如何通过 Hub API 管理组织成员角色和资源组成员身份：更改成员的组织角色和资源组分配、列出资源组、向组添加用户以及批处理工作流。

**目录：**

- [Change member role via API](#change-member-role-via-api) — 设置成员的组织角色和资源组分配（每个请求一个成员）。
- [Resource Groups API](#resource-groups-api) — 列出资源组并向其中添加用户。
- [Configure auto-join via API](#configure-auto-join-via-api) — 启用或禁用资源组上的自动加入。

---

## 通过API更改成员角色

您可以使用 Hub API 更改成员的**组织角色**（无访问权限/读取/贡献者/写入/管理员）以及他们在**资源组**中的角色（可选）。 API 更新**每个请求一个成员**。要更改多个成员的角色，请循环调用 API（示例如下）。

**OpenAPI 参考：** PUT /api/organizations/{name}/members/{username}/role

### 先决条件- 您的组织必须有**订阅计划**（例如团队或企业）。否则端点返回 402。
- 您必须通过身份验证，成为具有组织**写入**（或管理员）权限的组织成员。
- 目标用户必须已经是组织的**成员**。

### 基本 URL 和身份验证

- **基本网址：** `https://huggingface.co`
- **身份验证：** 在请求标头中发送您的令牌：
  ```http
  Authorization: Bearer <your_access_token>
  ```
  创建一个细粒度令牌，其“写入组织设置/成员管理的访问权限”权限范围为您的组织（[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)）。

### 更改成员角色端点

**请求**

```http
PUT /api/organizations/{org_name}/members/{username}/role
Authorization: Bearer <your_access_token>
Content-Type: application/json

{
  "role": "read",
  "resourceGroups": []
}
```- **路径参数**
  - `org_name`：组织段（例如`my-org`）。
  - `username`：拥抱您要更改角色的成员的 **用户名**。
- **身体**
  - `role`（必填）：成员的**组织级别**角色。以下之一：`"no_access"`、`"read"`、`"contributor"`、`"write"` 或 `"admin"`。
  - `resourceGroups`（可选）：该用户的资源组分配数组。每一项：
    - `id`：资源组 ID（24 个字符的十六进制字符串；从 [resource groups list API](#list-resource-groups) 获取 ID）。
    - `role`：该资源组中的角色：`"read"`、`"contributor"`、`"write"` 或 `"admin"`。
  - 如果省略 `resourceGroups` 或传递 `[]`，将从所有资源组中删除用户。要仅更改组织角色并保持资源组不变，请传递其当前资源组成员身份（正文始终设置组织角色和资源组列表）。

**示例 (curl) – 将组织角色设置为“读取”，无资源组（删除用户之前所在的任何资源组）**

```bash
curl -s -X PUT \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"role":"read","resourceGroups":[]}' \
  "https://huggingface.co/api/organizations/my-org/members/member1/role"
```

**示例 (curl) – 设置组织角色和资源组角色（覆盖任何当前组）**

```bash
curl -s -X PUT \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"role":"write","resourceGroups":[{"id":"507f1f77bcf86cd799439011","role":"read"}]}' \
  "https://huggingface.co/api/organizations/my-org/members/member2/role"
```

**成功响应：**状态`200 OK`；本体：`{ "success": true }`。

**典型错误**- `400` — 无效正文（例如无效的角色或资源组`id`）。
- `402` — 组织没有订阅计划。
- `403` — 不允许（例如，您在组织中缺少 Write，或者资源组不在组织中）。
- `404` — 未找到组织或用户。

### 更新多个成员

API 更改**每个请求一个成员**。没有批量端点。要更新许多成员，请为每个用户名调用一次端点（例如，从列表或 CSV）。

**示例：Bash – 循环用户名，所有角色相同**

```bash
ORG_NAME="my-org"
ROLE="read"
for username in member1 member2 member3 member4; do
  echo "Setting $username to $ROLE ..."
  curl -s -w "\n%{http_code}" -X PUT \
    -H "Authorization: Bearer $HF_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"role\":\"$ROLE\",\"resourceGroups\":[]}" \
    "https://huggingface.co/api/organizations/$ORG_NAME/members/$username/role"
  echo ""
done
```

**示例：Python – 循环用户名**

```python
import os
import requests

BASE_URL = "https://huggingface.co"
HF_TOKEN=REDACTED, "")

def change_member_role(org_name: str, username: str, role: str, resource_groups: list | None = None):
    payload = {"role": role, "resourceGroups": resource_groups or []}
    r = requests.put(
        f"{BASE_URL}/api/organizations/{org_name}/members/{username}/role",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        json=payload,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

org_name = "my-org"
role = "read"
for username in ["member1", "member2", "member3", "member4"]:
    print(f"Setting {username} to {role} ... ", end="")
    try:
        change_member_role(org_name, username, role)
        print("OK")
    except Exception as e:
        print(f"Failed: {e}")
```

对于每个用户的不同角色，循环遍历`(username, role)`对（例如来自CSV）并为每个调用`change_member_role`。

---

## 资源组 API

通过以下端点，您可以**列出**资源组并向其中**添加**用户。要**更改**现有成员的组织级角色或其资源组分配，请参阅上面的[Change member role via API](#change-member-role-via-api)。

**OpenAPI参考：** [Resource groups](https://huggingface.co/spaces/huggingface/openapi#tag/resource-groups)

**目录 - API 方法：**|目标|部分|
| -------------------------------------------------- | ----------------------------------------------------------------------------------- |
|将许多用户添加到**一个**资源组 | [Add users to a resource group](#add-users-to-a-resource-group) |
|将 **相同​​* 用户添加到 **许多** 资源组 | [Batch-add by looping over the API](#batch-add-by-looping-over-the-api) |
|为每个组添加 **不同** 用户 | [Batch-add by looping over the API](#batch-add-by-looping-over-the-api) |

### 基本 URL 和身份验证

- **基本网址：** `https://huggingface.co`
- **身份验证：** 使用以下之一：
  - **访问令牌（建议用于脚本）：** 创建一个细粒度令牌，其范围为您的组织（[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)）的“对组织设置/成员管理的写入访问权限”。在请求头中发送：
    ```http
    Authorization: Bearer <your_access_token>
    ```
  - **会话 cookie：** 如果从与 Hub UI 共享同一会话的浏览器或工具进行调用，则会自动发送 cookie。

### 列出资源组

获取您可以为组织管理的所有资源组。使用它来获取添加用户调用的每个组的 `id`。

**请求**

```http
GET /api/organizations/{org_name}/resource-groups
Authorization: Bearer <your_access_token>
```

**示例（卷曲）**

```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/organizations/my-org/resource-groups"
```

**响应示例（已修剪）**

```json
[
  {
    "id": "507f1f77bcf86cd799439011",
    "name": "Cohort 2024",
    "description": "Members in this group",
    "users": [...],
    "repos": [...]
  }
]
```

添加用户时使用每个资源组的`id`。### 将用户添加到资源组

在一个请求中将一名或多名用户添加到单个资源组。您可以在同一请求中发送多个用户。

**请求**

```http
POST /api/organizations/{org_name}/resource-groups/{resource_group_id}/users
Authorization: Bearer <your_access_token>
Content-Type: application/json

{
  "users": [
    { "user": "member1", "role": "read" },
    { "user": "member2", "role": "read" },
    { "user": "member3", "role": "write" }
  ]
}
```

- **路径参数**
  - `org_name`：组织段（例如`my-org`）。
  - `resource_group_id`：资源组的`id`（来自列表端点的 24 个字符的十六进制字符串）。
- **身体**
  - `users`：对象数组。每个对象必须具有：
    - `user`：拥抱脸**用户名**（必填）。
    - `role`：`"read"`、`"contributor"`、`"write"`、`"admin"` 之一。

**示例（卷曲）**

```bash
curl -s -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"users":[{"user":"member1","role":"read"},{"user":"member2","role":"read"}]}' \
  "https://huggingface.co/api/organizations/my-org/resource-groups/507f1f77bcf86cd799439011/users"
```

**成功：**状态`200 OK`； body 是更新后的资源组对象（包括`users`中的新用户）。

**典型错误：**

- `400` — 例如找不到用户、重复的用户名或无效的正文。
- `403` — 不允许（例如不在组织中，或已在资源组中）。该消息将指示用户是否不在组织中或已在组中。

### 通过电子邮件添加成员（解决方法）

添加用户端点仅接受 **Hugging Face 用户名**，而不接受电子邮件。如果您有**电子邮件**列表（例如会员电子邮件），您可以先解析电子邮件 → 用户名，然后调用添加用户 API。请注意，电子邮件过滤**仅**在电子邮件的域与组织允许的域之一匹配时起作用：**组织电子邮件域**（设置→帐户→组织电子邮件域）和/或组织的**SSO 允许的域**（如果配置了 SSO）。

**第 1 步 – 将电子邮件解析为用户名**

```http
GET /api/organizations/{org_name}/members?email={email}&limit=1
Authorization: Bearer <your_access_token>
```

响应是一个成员数组；每个成员都有`user`（用户名）。使用 `user` 进行添加用户调用。

**步骤 2 – 添加到资源组**

在正常的添加用户请求中使用步骤 1 中的用户名：

```http
POST /api/organizations/{org_name}/resource-groups/{resource_group_id}/users
Content-Type: application/json
Body: { "users": [{ "user": "<username from step 1>", "role": "read" }] }
```

**示例：一封电子邮件 (bash)**

```bash
ORG_NAME="my-org"
RG_ID="507f1f77bcf86cd799439011"
EMAIL="member@org.com"

# Step 1: look up member by email (domain must match org's Organization email domain or SSO allowed domains)
MEMBERS=$(curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/organizations/$ORG_NAME/members?email=$EMAIL&limit=1")
USERNAME=$(echo "$MEMBERS" | jq -r '(.[0] // {} | .user // "")')
if [ -z "$USERNAME" ]; then
  echo "No member found for $EMAIL"
  exit 1
fi
# Step 2: add to resource group
curl -s -X POST -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" \
  -d "{\"users\":[{\"user\":\"$USERNAME\",\"role\":\"read\"}]}" \
  "https://huggingface.co/api/organizations/$ORG_NAME/resource-groups/$RG_ID/users"
```

**示例：循环中的多封电子邮件 (Python)**

```python
import os
import requests

BASE = "https://huggingface.co"
ORG = "my-org"
RG_ID = "507f1f77bcf86cd799439011"
ROLE = "read"
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "Content-Type": "application/json"}

emails = ["member1@org.com", "member2@org.com"]
for email in emails:
    # Step 1: resolve email → username (email domain must match org's Organization email domain or SSO allowed domains)
    r = requests.get(f"{BASE}/api/organizations/{ORG}/members", params={"email": email, "limit": 1}, headers=headers)
    r.raise_for_status()
    members = r.json()
    if not members:
        print(f"No member found for {email}")
        continue
    username = members[0]["user"]
    # Step 2: add that user to the resource group
    add_r = requests.post(
        f"{BASE}/api/organizations/{ORG}/resource-groups/{RG_ID}/users",
        headers=headers,
        json={"users": [{"user": username, "role": ROLE}]},
    )
    if add_r.status_code == 200:
        print(f"Added {username} ({email})")
    else:
        print(f"Failed {email}: {add_r.status_code} {add_r.text}")
```

如果用户已经在资源组中，则添加调用返回`403`；该脚本将其报告为失败，如果您愿意，您可以跳过或忽略该情况。

**限制：** 仅当组织设置了 **组织电子邮件域** 和/或 **SSO 允许的域**，并且电子邮件的域与其中之一匹配时，电子邮件过滤器才适用。否则您无法通过会员API通过电子邮件进行查找；您需要另一个电子邮件来源→用户名（例如您自己的目录）。

### 通过循环 API 批量添加您可以在一个或几个请求中将许多用户添加到**一个**资源组（例如，对用户名列表进行分块），或者通过循环遍历组并为每个资源组调用添加用户端点将用户添加到**多个**资源组。

**示例：Bash – 一个请求中一组、多个用户**

```bash
#!/bin/bash
# Add a list of users to a single resource group.
# Usage: ./add-users-to-rg.sh <org_name> <resource_group_id> <role>

ORG_NAME="${1:-my-org}"
RG_ID="${2:-507f1f77bcf86cd799439011}"
ROLE="${3:-read}"

USERS="member1 member2 member3 member4"
USERS_JSON=$(echo "$USERS" | tr ' ' '\n' | while read u; do
  [ -n "$u" ] && echo "{\"user\":\"$u\",\"role\":\"$ROLE\"}"
done | paste -sd ',' -)

curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"users\":[$USERS_JSON]}" \
  "https://huggingface.co/api/organizations/$ORG_NAME/resource-groups/$RG_ID/users"
```

**示例：Bash – 循环多个组**

```bash
# Get group IDs and add users to each
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/organizations/my-org/resource-groups" \
  | jq -r '.[].id' \
  | while read -r RG_ID; do
      [ -z "$RG_ID" ] && continue
      echo "Adding users to resource group $RG_ID ..."
      curl -s -X POST -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" \
        -d "{\"users\":[$USERS_JSON]}" \
        "https://huggingface.co/api/organizations/my-org/resource-groups/$RG_ID/users"
    done
```

**示例：Python – 批量添加到一个或多个组**

```python
import os
import requests

BASE_URL = "https://huggingface.co"
HF_TOKEN=REDACTED, "")

def list_resource_groups(org_name: str):
    r = requests.get(
        f"{BASE_URL}/api/organizations/{org_name}/resource-groups",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
    )
    r.raise_for_status()
    return r.json()

def add_users_to_resource_group(org_name: str, resource_group_id: str, users_with_roles: list):
    """users_with_roles: list of {"user": "username", "role": "read"|"write"|"contributor"|"admin"}"""
    r = requests.post(
        f"{BASE_URL}/api/organizations/{org_name}/resource-groups/{resource_group_id}/users",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        json={"users": users_with_roles},
    )
    if r.status_code != 200:
        raise RuntimeError(f"Add users failed {r.status_code}: {r.text}")
    return r.json()

# Example: same users added to every resource group
org_name = "my-org"
role = "read"
usernames = ["member1", "member2", "member3"]
users_with_roles = [{"user": u, "role": role} for u in usernames]

for rg in list_resource_groups(org_name):
    add_users_to_resource_group(org_name, rg["id"], users_with_roles)
```

对于一长串用户名，将它们分块（例如每个请求 50 个）并为每个块调用一次 API 以避免大型请求体或超时。

### 重要提示1. **仅限用户名** — API 接受 Hugging Face **用户名**，而不接受电子邮件。在调用 API 之前，您需要从电子邮件→用户名（例如，从您的目录或组织成员列表）的映射。
2. **用户必须位于组织中** — 请求中的每个用户必须已经是组织的成员。否则，请求将返回 `403` 并显示某些用户不在组织中的消息。
3. **幂等性** — 如果用户已在资源组中，则后端可能会针对该请求返回`403`。如果您首先获取组的 `users` 列表，您的脚本可以捕获错误并继续，或者跳过组中已有的用户。
4. **速率限制** — 对于大批量，请考虑在请求之间添加较短的延迟（例如 0.5-1 秒），以避免达到速率限制。
5. **令牌范围** — 访问令牌必须对组织具有足够的权限（通常至少“对组织设置/成员管理的写入访问权限”）。安全地创建和存储令牌；不要将其提交给版本控制。

---

## 通过 API 配置自动加入[Auto-join](./security-resource-groups#auto-join) 自动将组织成员添加到指定角色的资源组。您可以通过 API 启用或禁用它，并可选择是否包含每个组织成员或仅包含 Read+ 成员。

**启用自动加入**

```http
POST /api/organizations/{org_name}/resource-groups/{resource_group_id}/settings
Authorization: Bearer <your_access_token>
Content-Type: application/json

{
  "autoJoin": {
    "enabled": true,
    "role": "read",
    "scope": "read_plus"
  }
}
```

- **路径参数**
  - `org_name`：组织团块（例如`my-org`）。
  - `resource_group_id`：资源组的 ID（24 个字符的十六进制字符串；从 [list resource groups endpoint](#list-resource-groups) 获取 ID）。
- **身体**
  - `role`：分配给自动添加的成员的角色。 `"read"`、`"contributor"`、`"write"` 或 `"admin"` 之一。
  - `scope`（可选）：自动添加哪些组织成员。使用 `"all"` 包含每个组织成员。使用 `"read_plus"` 排除具有 `no_access` 组织角色的成员。省略时默认为 `"all"`。

在现有资源组上启用自动加入会立即添加与所选范围匹配的当前组织成员（回填）。

**禁用自动加入**

使用`"enabled": false`发送相同的请求。禁用时不需要 `role` 字段：

```http
POST /api/organizations/{org_name}/resource-groups/{resource_group_id}/settings
Authorization: Bearer <your_access_token>
Content-Type: application/json

{
  "autoJoin": {
    "enabled": false
  }
}
```

> [!注意]
> 禁用自动加入不会删除之前自动加入的成员。它只会阻止自动添加未来的组织成员。现有成员仍保留在资源组中。### 在拥抱脸部使用 mlx-image
https://huggingface.co/docs/hub/mlx-image.md