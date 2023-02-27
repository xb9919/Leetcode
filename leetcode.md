# 目录 
- [目录](#目录)
- [leetcode 刷题记录](#leetcode-刷题记录)
  - [1. 两数相加](#1-两数相加)
  - [2. 无重复字符的最长子串](#2-无重复字符的最长子串)
  - [3. 寻找两个正序数组的中位数](#3-寻找两个正序数组的中位数)
  - [4. 最长回文子串](#4-最长回文子串)
  - [5. Z 字形变换](#5-z-字形变换)
  - [6. 整数反转](#6-整数反转)
  - [7. 字符串转换整数 (atoi)](#7-字符串转换整数-atoi)
  - [8. 回文数（简单）](#8-回文数简单)
  - [9. 盛最多水的容器](#9-盛最多水的容器)
  - [10. 整数转罗马数字](#10-整数转罗马数字)
  - [15. 三数之和](#15-三数之和)
  - [16. 最接近的三数之和](#16-最接近的三数之和)
  - [17. 最接近的三数之和](#17-最接近的三数之和)
  - [19. 删除链表的倒数第 N 个结点](#19-删除链表的倒数第-n-个结点)
  - [20. 有效的括号](#20-有效的括号)

# leetcode 刷题记录

## priority_queue

定义：**priority_queue<Type, Container, Functional>**

Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector,deque等等，但不能用 list。STL里面默认用的是vector），Functional 就是比较的方式。

```
//升序队列，小顶堆
priority_queue <int,vector<int>,greater<int> > q;
//降序队列，大顶堆
priority_queue <int,vector<int>,less<int> >q;

//greater和less是std实现的两个仿函数（就是使一个类的使用看上去像一个函数。其实现就是类中实现一个operator()，这个类就有了类似函数的行为，就是一个仿函数类了）
```



## 1. 两数相加
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

我的：
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        timesl1=1
        timesl2=1
        l1_v=0
        l2_v=0
        while True:
            if (not l1) and (not l2):
                break
            if l1:
                l1_v+=l1.val*timesl1
                timesl1*=10
                l1 = l1.next
            if l2:
                l2_v+=l2.val*timesl2
                timesl2*=10
                l2 = l2.next
        result = l1_v + l2_v
        timesl1=1
        a = ListNode(result%10)
        pre = a
        result = int(result/10)
        while result!=0:
            b = ListNode(result%10)
            pre.next = b
            pre = b  
            result = int(result/10)             
        return a
```
牛的
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        res = ListNode()
        dummy = ListNode()
        res.next = dummy
        flag = False
        while l1 or l2 or flag:
            summ = 0
            if l1: summ += l1.val
            if l2: summ += l2.val
            if flag: summ += 1
            if summ >= 10: 
                flag = True
                dummy.val = summ % 10
            else:
                flag = False
                dummy.val = summ
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            if l1 or l2 or flag:
                dummy.next = ListNode()
                dummy = dummy.next
        return res.next

```
主要区别在对于每一位，可以直接使用listnode对应位相加模10，同时以flag存储是否进位不用重构出两个数，并且

## 2. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

我的：
使用```unordered_set```类，注意```.find()```方法是返回一个迭代器，找不到就返回空迭代器，即```.end()```。如果不满足条件（当前元素在子串pre中已经存在了）就将前面的逐个擦除```.erase()```
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int len = 0;
        int result = 0;
        unordered_set<char> pre;
        int idx = 0;
        for (int i = 0; i < s.length(); i++) {
            while (pre.find(s[i]) != pre.end()) {
                pre.erase(s[idx]);
                idx++;
            }
            pre.insert(s[i]);
            len = pre.size();
            if (len > result)
                result = len;
        }
        if (len > result)
            result = len;
        return result;
    }
};
```
佬的：从当前位置i往前看，start记录的是和当前位置i最接近的满足题意的下标：
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
    int max=0,start=0,end=0;
	int n=s.size();
      for(int i=0;i<n;i++)
      {
          end=i;
        for( int j=start;j<i;j++)
	   {
	     if(s[i]==s[j])
		 {
		   start=j+1;
		   max=(max>end-start+1)?max:(end-start+1);
		   break;
		 }
	   }
		   max=(max>end-start+1)?max:(end-start+1);
      }
      return max;
    }
};
```
## 3. 寻找两个正序数组的中位数
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
算法的时间复杂度应该为 O(log (m+n))  (我的好像是O(M+N)？)。

难度标困难但实际比较简单：
```
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1 = (len(nums1)+len(nums2)+1)/2 if (len(nums1)+len(nums2))%2!=0 else (len(nums1)+len(nums2)+1)//2
        idx1,idx2 = 0,0
        flag=0
        for i in range(len1):
            if idx1>=len(nums1) or (idx2<len(nums2) and nums1[idx1]>nums2[idx2]):
                idx2+=1
                flag=2
            else:
                idx1+=1
                flag=1     
        if (len(nums1)+len(nums2))%2!=0:
            if flag==1:
                return nums1[idx1-1]
            else:
                return nums2[idx2-1]
        elif flag==1:
            if idx1>=len(nums1):
                return (nums1[-1]+nums2[idx2])/2.0
            elif idx2>=len(nums2):
                return (nums1[idx1-1]+nums1[idx1])/2.0
            else:
                return (nums1[idx1-1]+min(nums1[idx1],nums2[idx2]))/2.0
        else:
            if idx2>=len(nums2):
                return (nums2[-1]+nums1[idx1])/2.0
            elif idx1>=len(nums1):
                return (nums2[idx2-1]+nums2[idx2])/2.0
            else:
                return (nums2[idx2-1]+min(nums1[idx1],nums2[idx2]))/2.0
```
## 4. 最长回文子串
给你一个字符串 s，找到 s 中最长的回文子串。
示例 2：
```
输入：s = "cbbd"
输出："bb"
```
我的：这题有点坑，主要是对函数不熟悉，```s.substr(i-j, len2)```的参数2应该是长度，一直以为是下标
```
class Solution {
public:
    string longestPalindrome(string s) {
        int maxl = 1;
        string result = "";
        int maxl1 = 0;
        int maxl2 = 0;
        int FLAG = 0;
        int FLAG1 = 1;
        int len = 1;
        int len2 = 0;
        result = s[0];

        for (int i = 0; i < s.length()-1; i++) {
            maxl1 = i < (s.length() - i - 1) ? i : (s.length() - i - 1);
            maxl2 = i < (s.length() - i-2) ? i : (s.length() - i-2);
            FLAG = 0;
            FLAG1 = 1;
            len = 1;
            len2 = 0;
            if (s[i] == s[i + 1]) {
                FLAG = 1;
            }
            for (int j = 0; j <= maxl1; j++) {
                if ((s[i - j] == s[i + j])&& (FLAG1==1) && (j != 0)) {
                    len += 2;
                    if (len > maxl) {
                        result = s.substr(i - j, len);
                        maxl = len;
                    }
                }
                else if((s[i - j] != s[i + j])) FLAG1 = 0;
                if (FLAG == 1) {
                    if (j <=maxl2 && (s[i - j] == s[i + j + 1])) {
                        len2 += 2;
                        if (len2 > maxl) {
                            maxl = len2; 
                            result = s.substr(i-j, len2);
                        }
                    }
                    else if (s[i - j] != s[i + j + 1])FLAG = 0;
                if ((FLAG == 0) && (FLAG1 == 0)) break;   
                }
            }
        }
        return result;
    }
};
```
## 5. Z 字形变换
将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
```
P   A   H   N
A P L S I I G
Y   I   R
```
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
我的思路：根据和行取模，逐项算他的行数，或者是遍历元素时设置一个flag，为1时往下走，即对应行的string加这个字符（后者感觉要快一些）
```
class Solution {
public:
    string convert(string s, int numRows) {
        
        string out[numRows];
        string result="";
        int rows, _div;
        if( numRows==1) return s;
        for (int i = 0; i < s.length(); i++) {
            _div = i % (2 * numRows - 2);
            if (_div < numRows) rows = _div;
            else rows = numRows - (_div - numRows + 1)-1;//计算出当前是第几行的
            out[rows] = out[rows] + s[i];
        }
        for(int i = 0; i < numRows; i++) result+=out[i];
        return result;
    }
};
```
## 6. 整数反转
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
如果反转后整数超过 32 位的有符号整数的范围 [−2^31,  2^31 − 1] ，就返回 0。

感觉对python来说很简单，python无上限好像？难点在判断溢出吧
```
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x<-pow(2,31) or x>(pow(2,31)-1):
            return 0
        times = 1 if x>=0 else -1
        x = abs(x)
        tmp = []
        result=0
        while True:
            if x>=10:
                tmp.append(x%10)
                x=int(x/10)
            else:
                tmp.append(x)
                break
        t = pow(10,len(tmp)-1)# len(tmp)
        for tt in tmp:
            result+=t*tt
            t=t/10
        return times*result if -2147483648 < times*result < 2147483647 else 0
```
针对JAVA 溢出不会报错，可以判断临时的翻转结果，如果这个翻转结果除以10不等于上一个结果，说明有溢出
```
int tmp = res * 10 + x % 10;
if (tmp / 10 != res) { // 溢出!!!
    return 0; 
```
 c++是在中间判断INT_MIN/10和INT_MAX/10的大小关系更大就溢出？应该加个个位数的判断
 ```
 class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            if (rev < INT_MIN / 10 || rev > INT_MAX / 10) {
                return 0;
            }
            int digit = x % 10;
            x /= 10;
            rev = rev * 10 + digit;
        }
        return rev;
    }
};
 ```
## 7. 字符串转换整数 (atoi)
请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
函数 myAtoi(string s) 的算法如下：
读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：
本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符

题目不难 但是考虑的情况不少，很难一次考虑全（符号只能读一次。连续的符号第二个是作为字符了，且首个不能是字母）
```
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        Flag = True
        times=1
        result=0
        tFlag = True
        for s1 in s:
            try:
                #res = int(s1)
                # if s1!=" " and tFlag:
                #     Flag = False
                res = int(s1)

                if result>214748364 or (result==214748364 and res>=8):
                    if times==1:
                        return 2147483647
                    else:
                        return -2147483648 
                result = 10*result+res
                Flag = False
            except:
                if not Flag:
                    return times*result
                if s1=="-" and tFlag:
                    times=-1
                    tFlag = False
                elif s1=="+" and tFlag:
                    tFlag = False
                elif s1==" " and tFlag:
                    continue
                else:
                    return times*result
        return times*result
```
最快的都是正则表达式的（不会） 佬的：
"""
import re
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        INT_MAX = 2147483647    
        INT_MIN = -2147483648
        str = s.lstrip()      #清除左边多余的空格
        num_re = re.compile(r'^[\+\-]?\d+')   #设置正则规则
        num = num_re.findall(str)   #查找匹配的内容
        num = int(*num) #由于返回的是个列表，解包并且转换成整数
        return max(min(num,INT_MAX),INT_MIN)    #返回值
"""
般般快的,思路不难，应该想得到才对。。。
```
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        string=s.strip()
        flag=True
        ans=0
        for index,item in enumerate(string):
            if index==0 and item =='-':
                flag=False
            elif index==0and item =='+':
                flag=True
            elif '9'<item or item < '0':
                break
            else:
                ans=ans*10+(ord(item)-ord('0'))
        ans= ans if flag else -ans
        ans=-2**31 if ans<-2**31 else ans
        ans=2**31-1 if ans>=2**31 else ans
        return ans
```

## 8. 回文数（简单）
给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
```
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        if (x%10 == 0 and x!=0) return false;
        string x1 = to_string(x);
        bool odded = x1.length() % 2 == 0 ? false : true;
        int idx = (x1.length()) / 2;
        if (odded) {
            
            for (int i = 0; i <= idx;i++) {
                if (x1[idx - i] == x1[idx + i]) continue;
                else return false;
            }
        }
        else {
            idx--;
            for (int i = 0; i <= idx; i++) {
                if (x1[idx-i] == x1[idx +1+i]) continue;
                else return false;
            }
        }
        return true;
    
    }
};
```
## 9. 盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。

双指针经典，暴力法超时：
```
    int maxArea(vector<int>& height) {
        int max_ = min(height[0], height[1]);
        int area = 0;
        int pre = height[0];

        for (int i = 1; i < height.size(); i++) {
            
            if (abs(height[i] - pre) < 1) continue;
            for (int j = 0; j < i; j++) {
                //h = height[i]<
                if (height[j] < pre) continue;
                pre = min(height[i], height[j]);
                area = pre * (i - j);
                max_ = max(area, max_);
            }
        }
        return max_;
    }
```
```
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                max(res, (j - i) * height[i++]) :
                max(res, (j - i) * height[j--]);
        }
        return res;
    }
};
```
## 10. 整数转罗马数字

最简单的题了应该是
执行用时：4 ms, 在所有 C++ 提交中击败了81.30%的用户
内存消耗：5.7 MB, 在所有 C++ 提交中击败了86.62%的用户

```cpp
class Solution {
public:
    string intToRoman(int num) {
        string a = "";
        int x = 0;
        int i = 0;
        char label[] = { 'I','V','X','L','C','D','M' };
        while (true) {
            string tmp = "";
            x = num % 10;
            if (x < 4) {
                for (int j = 0; j < x; j++) tmp += label[i];
            }else if (x == 4) {
                tmp += label[i];
                tmp += label[i + 1];
            }else if (x == 9) {
                tmp += label[i];
                tmp += label[i + 2];
            }else {
                tmp += label[i + 1];
                for (int j = 0; j < (x - 5); j++) tmp += label[i];
            }
            i += 2;
            num = num / 10;
            a = tmp + a;
            if (num == 0) break;
        }
        return a;
    }
};
```
## 15. 三数之和
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

难点在要在 $O(n^2)$解决，以及去重
```
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        int L;
        int R = nums.size() - 1;
        if (R < 2) return result;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); i++) {
            if(i>0&&nums[i]==nums[i-1]) continue;
            L = i + 1; R = nums.size() - 1;
            while (L < R) {
                if (nums[i] + nums[L] + nums[R] == 0) {
                    result.push_back({ nums[i],nums[L],nums[R] });
                    L++; R--;
                    while (L < R && nums[L] == nums[L - 1]) L++;
                    while (L < R && nums[R] == nums[R + 1]) R--;
                }
                else if (nums[i] + nums[L] + nums[R] < 0) L++;
                else R--;
            }
        }
        return result;
    }
};
```

## 16. 最接近的三数之和
给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与 target 最接近。
返回这三个数的和。假定每组输入只存在恰好一个解。
```
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int result, L, R;
        int gap = INT_MAX;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); i++) {
            L = i + 1;
            R = nums.size() - 1;
            if (i != 0 && nums[i] == nums[i - 1]) continue;
            while (L < R) {
                if ((nums[i] + nums[L] + nums[R]) < target) {
                    if (abs(nums[i] + nums[L] + nums[R] - target) < gap) {
                        result = nums[i] + nums[L] + nums[R];
                        gap = abs(result - target);
                    }
                    L++;
                }
                else if ((nums[i] + nums[L] + nums[R]) == target) return target;
                else {
                    if (abs(nums[i] + nums[L] + nums[R] - target) < gap) {
                        result = nums[i] + nums[L] + nums[R];
                        gap = abs(result - target);
                    }
                    R--;
                }
            }
        }
        return result;
    }
};
```
## 17. 最接近的三数之和
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

还是对cpp不熟悉，debug de了很久用的递归，回溯会快很多
注：to_string(char)是他的ascii码,
```
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> map = { "","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz" };
        vector<string> result = {};
        if (digits.size() == 0) return result;
        else if (digits.size() == 1) {
            int number = digits[0] - '0';
            string tmap = map[number - 1];
            for (int i = 0; i < map[number - 1].size(); i++) {
                cout << map[number - 1][i];
                string t1(1, map[number - 1][i]);
                result.push_back(t1);
            }
            return result;
        }
        int number = digits[0] - '0';
        for (int j = 0; j < map[number - 1].size(); j++) {
            auto tmp = letterCombinations(digits.substr(1));
            for (auto t = tmp.begin(); t != tmp.end(); t++)
                result.push_back((*t).insert(0,1, map[number - 1][j]));
        }
        return result;
    }
};
```
```
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        vector<string> combinations;
        if (digits.empty()) {
            return combinations;
        }
        unordered_map<char, string> phoneMap{
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };
        string combination;
        backtrack(combinations, phoneMap, digits, 0, combination);
        return combinations;
    }

    void backtrack(vector<string>& combinations, const unordered_map<char, string>& phoneMap, const string& digits, int index, string& combination) {
        if(index==digits.size()){
            combinations.push_back(combination);
        }else{
            string letter = phoneMap.at(digits[index]);
            for(int i=0;i<letter.size();i++){
                combination.push_back(letter[i]);
                backtrack(combinations,phoneMap,digits,index+1,combination);
                combination.pop_back();
            }
        }
        
    }
};
```
## 19. 删除链表的倒数第 N 个结点
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

只遍历一遍，双指针（对指针不熟啊）
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        int count=1;
        ListNode* head1 =head;
        ListNode* pre = head;
        ListNode* target = head;
        if(!head->next) return head->next;
        while(head1->next){
            head1 = (head1->next);
            count+=1;
            if(count>n){
                if(pre!=target){
                    pre=(pre->next);
                    target = (pre->next);
                }else{
                    target = pre->next;
                }
            }
        }
        if(target==head) return head->next;
        pre->next = target->next; 
        return head;
    }
};
```

## 20. 有效的括号
python经典白给题
```c++
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        maps={"(":")","[":"]","{":"}"}
        #left = ["(","[","{"}]
        #right = [")","]","}"]
        stack = []
        for it in s:
            if it in maps.keys():
                stack.append(it)
            else: 
                if len(stack)==0 or it!=maps[stack.pop()]:
                    return False
        if len(stack)==0:
            return True
        else:
            return False
 
 栈:
 class Solution {
public:
    bool isValid(string s) {
        stack<char> res;
        unordered_map<char,char> map;
        map['(']=')';
        map['[']=']';
        map['{']='}';
        for(int i=0;i<s.size();i++){
            if(s[i]=='('||s[i]=='['||s[i]=='{')
                res.push(s[i]);
            else{
                if(res.empty()) return false;
                char tmp = res.top();
                if((tmp=='}')||(tmp==']')||(tmp==')')||(s[i]!=map[tmp])) 
                    return false;
                res.pop();
            }
        }
        if(res.empty()) return true;
            return false;
    }
};

```
# 数组
## 二分查找
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left=0;
        int right=nums.size()-1;
        int middle;
        while(left<=right){
            middle = ((right-left)/2)+left;
            if(nums[middle]>target){
                right=middle-1;
            }else if(nums[middle]<target){
                left = middle+1;
            }else{
                return middle;
            }
        }
        return -1;
    }
};

```
## 移除元素
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并原地修改输入数组。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
示例 1: 给定 nums = [3,2,2,3], val = 3, 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。 你不需要考虑数组中超出新长度后面的元素。

示例 2: 给定 nums = [0,1,2,2,3,0,4,2], val = 2, 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
你不需要考虑数组中超出新长度后面的元素。

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int len=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]==val){
                len++;
            }
            else{
                nums[i-len] = nums[i];
            }
        }
        return nums.size()-len;
    }
};
```

##  有序数组的平方（双指针）

给你一个按 **非递减顺序** 排序的整数数组 `nums`，返回 **每个数字的平方** 组成的新数组，要求也按 **非递减顺序** 排序。

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int idx = nums.size()-1;
        vector<int> result(nums.size(),0);
        for(int i=0,j=nums.size()-1;i<=j;){
            if(nums[i]*nums[i]<nums[j]*nums[j]){
                result[idx--]=nums[j]*nums[j];
                --j;
            }
            else{
                result[idx--]=nums[i]*nums[i];
                i++;
            } 
        }
        return result;
    }
};
```

## [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)(滑动窗口，没做出来。。)

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int result = INT32_MAX;
        int sum = 0; 
        int i = 0; 
        int subLength = 0; 
        for (int j = 0; j < nums.size(); j++) {
            sum += nums[j];          
            while (sum >= s) {
                subLength = (j - i + 1); //第i个开始的最短大于0的连续长度
                result = result < subLength ? result : subLength;
                sum -= nums[i++]; //去掉第i个
            }
        }
        return result == INT32_MAX ? 0 : result;
    }
};
```

## [59. 螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)（暴力）

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> result(n,vector<int>(n,0));
        int loop=n/2;
        int num=1;
        
        for(int i=0;i<loop;i++){
            int up,row,col;
            for(up=i;up<n-i;up++){
                result[i][up] = num;
                num+=1;
            }
            for(row=i+1;row<n-i;row++){
                result[row][up-1] = num;
                num++;
            }
            for(col=up-2;col>=i;col--){
                result[row-1][col] = num;
                num++;
            }
            for(int j=row-2;j>i;j--){
                result[j][col+1] = num;
                num++;
            }  
        }
        if(n%2==1){
            result[n/2][n/2] = n*n;
        }
        return result;
    }
};
```

# 链表

## [203. 移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/) 

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

```c++
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if(!head) return head;
        while(head&&head->val==val) head = head->next;//找head
        if(!head) return head;
        ListNode* pre = head;
        ListNode* tmp = pre->next;
        while(tmp){
            if(tmp->val == val){
                pre->next = tmp->next;
                tmp = tmp->next;
            }else{
                pre = tmp;
                tmp = pre->next;
            }
            
        }
        return head;
    }
};
```

## [707. 设计链表](https://leetcode.cn/problems/design-linked-list/)

设计链表的实现。您可以选择使用单链表或双链表。单链表中的节点应该具有两个属性：val 和 next。val 是当前节点的值，next 是指向下一个节点的指针/引用。如果要使用双向链表，则还需要一个属性 prev 以指示链表中的上一个节点。假设链表中的所有节点都是 0-index 的。

在链表类中实现这些功能：

* get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
* addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
* addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
* addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，则不会插入节点。
* 如果index小于0，则在头部插入节点。
  deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。

```c++
class MyLinkedList {
    struct ListNode {
        int val;
        ListNode* next = nullptr;
        ListNode(int val) : val(val), next(nullptr) {};
    };
private:
    int _size;
    ListNode* _head;//= new ListNode(0);
public:
    MyLinkedList() {
        _size = -1;
        _head = nullptr;//new ListNode(0);
    }

    int get(int index) {
        if (index > _size || !_head || index < 0)   return -1;
        ListNode* pre = _head;
        while (index > 0) {
            index -= 1;
            pre = pre->next;
        }return pre->val;
    }

    void addAtHead(int val) {
        
        ListNode* n = new ListNode(val);
        _size++;
        if (!_head)  _head = n;
        else {
            n->next = _head;
            _head = n;
        }
    }

    void addAtTail(int val) {
        ListNode* n = new ListNode(val);
        ListNode* tmp = _head;
        _size++;
        if (!_head)  _head = n;
        else {
            while (tmp->next) tmp = tmp->next;
            tmp->next = n;
        }
    }

    void addAtIndex(int index, int val) {

        if (index > (_size+1)) return;
        else if (index == (_size+1)) {
            addAtTail(val);
            return;
        } 
        else if (index < 0) {
            addAtHead(val);
            return;
        }
        ListNode* n = new ListNode(val);
        ListNode* pre = _head;
        if (index == 0) {
            n->next = _head;
            _head = n;
             _size++;
            return;
        }
        index -= 1;
        while (index > 0) {
            index -= 1;
            pre = pre->next;
        }
        n->next = pre->next;//(pre->next)->next;
        pre->next = n;
        _size++;

    }

    void deleteAtIndex(int index) {
        if (index<0 || index>_size) return;
        _size--;
        ListNode* n = _head;
        if (index == 0) {
            _head = _head->next;
            return;
        }
        index -= 1;
        while (index > 0) {
            index -= 1;
            n = n->next;
        }
        n->next = (n->next)->next;
        
    }
};
```

## [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = head;
        if((!pre)||(!pre->next)) return pre;
        ListNode* n = pre->next;
        pre->next = nullptr;
        ListNode* tmp = n->next;
        while(tmp){
            n->next = pre;
            pre = n;
            n = tmp;
            tmp = tmp->next;
        }
        n->next = pre;
        return n;
    }
};
```

## [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)(快慢指针，虚拟头结点)

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        
        if(!head||(!head->next)) return head;
        ListNode* pre = new ListNode(0);
        pre->next = head;
        ListNode* n = pre;
        
        while((n->next)&&(n->next->next)){
            ListNode* tmp = n->next;
            ListNode* tmp1 = n->next->next;  
            n->next = tmp1;
            tmp->next = tmp1->next;
            tmp1->next = tmp;
            n = n->next->next;
        }     
        return pre->next;    
    }
};
```

## [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        int count=1;
        ListNode* head1 =head;
        ListNode* pre = head;
        ListNode* target = head;
        if(!head->next) return head->next;
        while(head1->next){
            head1 = (head1->next);
            count+=1;
            if(count>n){
                if(pre!=target){
                    pre=(pre->next);
                    target = (pre->next);
                }else{
                    target = pre->next;
                }
            }
        }
        if(target==head) return head->next;
        pre->next = target->next; 
        return head;
    }
};
```

## [面试题 02.07. 链表相交](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/)

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。

**要先算两个的长度然后倒推，如果有同一节点，该节点后面的长度是一样的**

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int len1=0;
        int len2=0;
        ListNode* a = headA;
        ListNode* b = headB;
        ListNode* Largre;
        ListNode* small;
        ListNode* res=NULL;
        while(a){
            a = a->next;
            len1++;
        }
        while(b){
            b = b->next;
            len2++;
        }
        if(len1>len2){
            Largre = headA;
            small = headB;
        }else{
            Largre = headB;
            small = headA;
        }
        for(int i=0;i<abs(len1-len2);i++){
            Largre = Largre->next;
        }
        while(Largre&&small){
            if(Largre==small){
                res = Largre;
                break;
            }
            Largre = Largre->next;
            small = small->next;
        }
        return res;
    }
};
```

## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。不允许修改链表。

也可以用快慢指针，快指针比慢指针先走多少步，看有没重叠的，重叠后就新建一个index1在当前节点,index2从头节点一起出发，相遇的地方就算环的入口。[公式推导](https://github.com/youngyangyang04/leetcode-master/blob/master/problems/0142.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8II.md)



```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        vector<ListNode*> res;
        ListNode* tmp = head;
        ListNode* r=NULL;       
        while(tmp){
            if(res.size()!=0){
                for(vector<ListNode*>::iterator it=res.begin();it!=res.end();it++){
                    if(*it==tmp){
                        r = tmp;
                        break;
                    }
                }
            }
            res.push_back(tmp);
            tmp = tmp->next;
            if(r) break;
        }
        return r;
    }
};
```

# 哈希

## [242. 有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        unordered_map<char,int> map;
        if(s.size()!=t.size()) return false;
        for(int i=0;i<s.size();i++){
            map[s[i]]++;
            map[t[i]]--;
        }
        for(unordered_map<char,int>::iterator it=map.begin();it!=map.end();it++){
            if(it->second!=0) return false;
        }
        return true;
    }
};
```

## [1002. 查找共用字符](https://leetcode.cn/problems/find-common-characters/)

给你一个字符串数组 words ，请你找出所有在 words 的每个字符串中都出现的共用字符（ 包括重复字符），并以数组形式返回。你可以按 任意顺序 返回答案。

要算字符在各个串里面最少的

```c++
class Solution {
public:
    vector<string> commonChars(vector<string>& words) {
        vector<string> res;
        unordered_map<char,int> map;
        for(int i=0;i<words[0].size();i++) map[words[0][i]]++;
        
            
        for(auto it=map.begin();it!=map.end();it++){
            for(int i=0;i<words.size();i++){
                int countM=0;
                for(int j=0;j<words[i].size();j++){
                    if(words[i][j]==it->first) countM++;
                }
                if(countM<it->second) it->second=countM;
            }
        }
        string s1;
        for(auto it=map.begin();it!=map.end();it++){
            for(int i=0;i<it->second;i++){
                s1 = it->first;
                res.push_back(s1);
            }
        }
        return res;
    }
};
```



```c++
class Solution {
public:
    vector<string> commonChars(vector<string>& A) {
        vector<string> result;
        if (A.size() == 0) return result;
        int hash[26] = {0}; // 用来统计所有字符串里字符出现的最小频率
        for (int i = 0; i < A[0].size(); i++) { // 用第一个字符串给hash初始化
            hash[A[0][i] - 'a']++;
        }

        int hashOtherStr[26] = {0}; // 统计除第一个字符串外字符的出现频率
        for (int i = 1; i < A.size(); i++) {
            memset(hashOtherStr, 0, 26 * sizeof(int));
            for (int j = 0; j < A[i].size(); j++) {
                hashOtherStr[A[i][j] - 'a']++;
            }
            // 更新hash，保证hash里统计26个字符在所有字符串里出现的最小次数
            for (int k = 0; k < 26; k++) {
                hash[k] = min(hash[k], hashOtherStr[k]);
            }
        }
        // 将hash统计的字符次数，转成输出形式
        for (int i = 0; i < 26; i++) {
            while (hash[i] != 0) { // 注意这里是while，多个重复的字符
                string s(1, i + 'a'); // char -> string
                result.push_back(s);
                hash[i]--;
            }
        }

        return result;
    }
};
```

## [349. 两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)

给定两个数组 `nums1` 和 `nums2` ，返回 *它们的交集* 。输出结果中的每个元素一定是 **唯一** 的。我们可以 **不考虑输出结果的顺序** 。

```c++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int,int> map;
        vector<int> res;
        for(int i=0;i<nums1.size();i++) map[nums1[i]]=0;
        for(auto it=map.begin();it!=map.end();it++) 
            for(int i=0;i<nums2.size();i++){
                if(nums2[i]==it->first) it->second=1;
            }
        for(auto it=map.begin();it!=map.end();it++)
            if(it->second!=0) res.push_back(it->first);
        return res;
    }
};
```

## [202. 快乐数](https://leetcode.cn/problems/happy-number/)

编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」 定义为：

对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
如果这个过程 结果为 1，那么这个数就是快乐数。

```c++
class Solution {
    unordered_set<int> set1;
public:
    bool isHappy(int n) {
        int sum=0;
        
        while(n>0){
            sum=(n%10)*(n%10)+sum;
            n = n/10;
        }
        if(set1.find(sum)!=set1.end()) return false;
        set1.insert(sum);
        if(sum==1) return true;
        
        return isHappy(sum);
        
    }
};
```

## [1. 两数之和](https://leetcode.cn/problems/two-sum/)

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

可以用哈希O(n)

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        unordered_map<int,int> map;
        for(int i=0;i<nums.size();i++){
            unordered_map<int,int>::iterator it =  map.find(target-nums[i]);
            if(it!=map.end()){
                res.push_back(i);
                res.push_back(it->second);
                break;
            }
            map[nums[i]] = i;
        }
        return res;
    }
};
```

## [454. 四数相加 II](https://leetcode.cn/problems/4sum-ii/)

给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：

0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0

```c++
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
        unordered_map<int,int> map;//(sum:times)

        for(int i=0;i<nums1.size();i++){
            for(int j=0;j<nums2.size();j++){
                map[nums1[i]+nums2[j]]+=1;
            }
        }
        int count=0;
        for(int i=0;i<nums3.size();i++){
            for(int j=0;j<nums4.size();j++){
                auto it = map.find(0-nums3[i]-nums4[j]);
                if(it!=map.end()) count+=it->second;
            }
        }
        return count;
    }
};
```

## [383. 赎金信](https://leetcode.cn/problems/ransom-note/)

给你两个字符串：ransomNote 和 magazine ，判断 ransomNote 能不能由 magazine 里面的字符构成。

如果可以，返回 true ；否则返回 false 。

magazine 中的每个字符只能在 ransomNote 中使用一次。

```c++
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char,int> map;
        for(int i=0;i<magazine.size();i++) map[magazine[i]]++;
        for(int i=0;i<ransomNote.size();i++){
            auto it = map.find(ransomNote[i]);
            if(it==map.end()||(it->second==0)) return false;
            it->second--;
        }
        return true;
    }
};
```

## [15. 三数之和(哈希或双指针)](https://leetcode.cn/problems/3sum/)

给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        //unordered_set<vector<int>> tmp;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++){
            if(nums[i]>0) break;
            if(i>0&&nums[i]==nums[i-1]) continue;
            unordered_set<int> set;
            for(int j=i+1;j<nums.size();j++){
                
                if(j<nums.size()-2&&nums[j]==nums[j+1]&&nums[j]==nums[j+2])
                    continue;
                auto it = set.find(0-nums[i]-nums[j]);
                if(it!=set.end()){
                    result.push_back({nums[i],nums[j],*it});
                    set.erase(0-nums[i]-nums[j]);
                }else{
                    set.insert(nums[j]);
                }
            }
        }
        return result;
    }
};
//双指针法
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        //unordered_set<vector<int>> tmp;
        sort(nums.begin(),nums.end());
        if(nums[0]+nums[1]>0) return result;
        for(int i=0;i<nums.size();i++){
            int L=i+1;
            int R = nums.size()-1;
            if(i>0&&nums[i]==nums[i-1]) continue;
            while(L<R){
                if(nums[i]+nums[L]+nums[R]>0) R--;
                else if(nums[i]+nums[L]+nums[R]<0) L++;
                else{
                    result.push_back({nums[i],nums[L],nums[R]});
                    L++;R--;
                    while(L<R&&nums[L]==nums[L-1]) L++;
                    while(L<R&&nums[R]==nums[R+1]) R--;
                }

            }
        }
        return result;
    }
};
```

# 字符串

## [541. 反转字符串 II](https://leetcode.cn/problems/reverse-string-ii/)

```c++
class Solution {
public:
    string reverseStr(string s, int k) {
        int count = 0;
        int i;
        for (i = 0; i < s.size(); i++) {
            count++;
        if (s.size() < 2 * k && s.size()>= k) {
            reverse(s.begin(), s.begin() + k);
            return s;
        }
        else if (s.size() < k) {
            reverse(s.begin(), s.end());
            return s;
        }
            if (count % (2 * k) == 0) {
                reverse(s.begin() + count - 2 * k, s.begin() + count - k);

                if (s.size() - count<(2 * k) && s.size() - i>=k) {
                    reverse(s.begin() + i+1, s.begin() + i + k+ 1);
                    break;
                }
                else if (s.size() - count < (2 * k) && s.size() - i < k) {
                    reverse(s.begin() + i+1, s.end());
                    break;
                }
            }
            
        }
        return s;
    }
};
```

## [剑指 Offer 05. 替换空格](https://leetcode.cn/problems/ti-huan-kong-ge-lcof/)

```c++
class Solution {
public:
    string replaceSpace(string s) {
        int count=0;
        int old = s.size();
        int i;
        for(i=0;i<s.size();i++) 
            if(s[i]==' ') count++;
        s.resize(s.size()+2*count);
        int new1 = s.size();
        int j=new1-1;
        for(i=old-1;i>=0;i--){
            if(s[i]==' '){
                s[j]='0';
                s[j-1]='2';
                s[j-2]='%';
                j-=3;
            }else{
              s[j]=s[i];
              j--;  
            }
        }
        return s;

    }
};
```

## [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/) 有点难

难在不使用O(n)的新内存

```c++
class Solution {
private:
    void convert(string& s,int i, int j){
        
        for(int k=0;k<=(i+j)/2-i;k++){
            swap(s[i+k],s[j-k]);
        }
    }
public:
    string reverseWords(string s) {
        int flag=1;
        int size=s.size();        
        for(int i=0;i<size;){
            if(flag==1&&s[i]==' '){
                 s = s.substr(1,s.size()-1);
                 size--;
                 continue;
            }
            flag=0;
            if(i>0&&s[i]==s[i-1]&&s[i]==' '){
                s.erase(s.begin()+i-1,s.begin()+i);
                size--;
                continue;
            }
            i++;
        }
        if(s[size-1]==' ') s = s.substr(0,s.size()-1);
        
        for(int i=0;i<s.size()/2;i++){
            swap(s[i],s[s.size()-1-i]);
        }
        int start=0;
        for(int i=0;i<s.size();i++){
            if(s[i]==' '){
                convert(s,start,i-1);
                start=i+1;
            }
        }
        convert(s,start,s.size()-1);
        return s;
    }
};

//效率更高的清洗
class Solution {
private:
    void clean(string&s){
        int slow=0;
        for(int i=0;i<s.size();i++){
            if(slow!=0&&s[i]!=' ') s[slow++]=' ';
            while(i<s.size()&&s[i]!=' ')s[slow++]=s[i++];
        }
        s.resize(slow);
    }
    void convert(string&s,int start,int end){
        for(int i=0;i<=(end-start)/2;i++){
            swap(s[i+start],s[end-i]);
        }
    }
public:
    string reverseWords(string s) {
        clean(s);
        for(int i=0;i<s.size()/2;i++){
            swap(s[i],s[s.size()-i-1]);
        }
        int start=0;
        for(int i=0;i<s.size();i++){
            if(s[i]==' '){
                convert(s,start,i-1);
                start = i+1;
            }
        }
        if(start<s.size()){
            convert(s,start,s.size()-1);
        }
        return s;
    }
};
```

## [剑指 Offer 58 - II. 左旋转字符串](https://leetcode.cn/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"

```
class Solution {
private:
    void convert(string& s, int start, int end){
        for(int i=0;i<=(end-start)/2;i++){
            swap(s[i+start],s[end-i]);
        }
    }
public:
    string reverseLeftWords(string s, int n) {
        if(n==0) return s;
        convert(s,0,s.size()-1);
        convert(s,0,s.size()-n-1);
        convert(s,s.size()-n,s.size()-1);
        return s;
    }
};
```

## [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)(KMP 还没做 有点难)

```
```



## [459. 重复的子字符串](https://leetcode.cn/problems/repeated-substring-pattern/)

给定一个非空的字符串 `s` ，检查是否可以通过由它的一个子串重复多次构成。

直接暴力

```c++
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int i,j;  
        for(i=0;i<s.size()/2;i++){
            string rot = s.substr(0,i+1);
            if(s.size()%(i+1)!=0) continue;
            int flag = 1;
            for(j=i+1;j<s.size();j=j+i+1){
                string comp = s.substr(j,i+1);
                if(comp!=rot){
                    flag = 0;
                    break;
                }
            }
            if(j>=s.size()&&flag==1) return true;
        }
        return false;
    }
};
```

# 栈

## [相关知识](https://programmercarl.com/%E6%A0%88%E4%B8%8E%E9%98%9F%E5%88%97%E6%80%BB%E7%BB%93.html#%E6%B1%82%E5%89%8D-k-%E4%B8%AA%E9%AB%98%E9%A2%91%E5%85%83%E7%B4%A0)

## [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：

实现 MyQueue 类：

void push(int x) 将元素 x 推到队列的末尾
int pop() 从队列的开头移除并返回元素
int peek() 返回队列开头的元素
boolean empty() 如果队列为空，返回 true ；否则，返回 false
说明：

你 只能 使用标准的栈操作 —— 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

```c++
class MyQueue {

private:
    stack<int> inStack;
    stack<int> outStack;
public:
    MyQueue() {

    }
    
    void push(int x) {
        inStack.push(x);
    }
    int res;
    int pop() {
        if(outStack.empty()){
            while(!inStack.empty()){
                res = inStack.top();
                outStack.push(res);
                inStack.pop();
                
            }      
        }
        res = outStack.top();
        outStack.pop();
        return res;
    }
    
    int peek() {
        int res;
        res = this->pop();
        outStack.push(res);
        return res;
    }
    
    bool empty() {
        if(inStack.empty()&&outStack.empty()) return true;
        return false;
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

## [225. 用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通栈的全部四种操作（push、top、pop 和 empty）。

实现 MyStack 类：

void push(int x) 将元素 x 压入栈顶。
int pop() 移除并返回栈顶元素。
int top() 返回栈顶元素。
boolean empty() 如果栈是空的，返回 true ；否则，返回 false 。


注意：

你只能使用队列的基本操作 —— 也就是 push to back、peek/pop from front、size 和 is empty 这些操作。
你所使用的语言也许不支持队列。 你可以使用 list （列表）或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。

```c++
class MyStack {
private:
    queue<int> que1;
    queue<int> que2;
public:
    MyStack() {

    }
    void push(int x) {
        que1.push(x);
    }
    int pop() {
        int res;
        while(que1.size()!=1){
            res = que1.front();
            que2.push(res);
            que1.pop();
        }
        res = que1.front();
        que1.pop();
        while(!que2.empty()){
            int tmp = que2.front();
            que2.pop();
            que1.push(tmp);
        }
        return res;
    }
    
    int top() {
                int res;
        while(que1.size()!=1){
            res = que1.front();
            que2.push(res);
            que1.pop();
        }
        res = que1.front();
        que1.pop();
        que2.push(res);
        while(!que2.empty()){
            int tmp = que2.front();
            que1.push(tmp);
            que2.pop();
        }
        return res;
    }
    
    bool empty() {
        return que1.empty();
    }
};
```

## [1047. 删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

```c++
class Solution {
public:
    string removeDuplicates(string s) {
        stack<char> st;
        for(int i=0;i<s.size();i++){
            if(!st.empty()){
                if(st.top()==s[i]){
                    st.pop();
                    continue;
                }
            }
            st.push(s[i]);
        }
        string res;
        while(!st.empty()){
            res+=st.top();
            st.pop();
        }
        reverse(res.begin(),res.end());
        return res;
    }
};
```

## [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

请你计算该表达式。返回一个表示表达式值的整数。

注意：

有效的算符为 '+'、'-'、'*' 和 '/' 。
每个操作数（运算对象）都可以是一个整数或者另一个表达式。
两个整数之间的除法总是 向零截断 。
表达式中不含除零运算。
输入是一个根据逆波兰表示法表示的算术表达式。
答案及所有中间计算结果可以用 32 位 整数表示。

```c++
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> st;
        for(int i=0;i<tokens.size();i++){
            if(tokens[i]=="+"||tokens[i]=="-"||tokens[i]=="*"||tokens[i]=="/"){
                int num1= st.top();
                st.pop();
                int num2 = st.top();
                st.pop();
                if (tokens[i] == "+") st.push(num2 + num1);
                if (tokens[i] == "-") st.push(num2 - num1);
                if (tokens[i] == "*") st.push(num2 * num1);
                if (tokens[i] == "/") st.push(num2 / num1);
            }else{
                st.push(stoi(tokens[i]));
            }
        }
        int result = st.top();
        return result;
    }
};
```

## [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)(用deque)

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

```c++
class Solution {
    deque<int> que;

    void push(int value){
        while(!que.empty()){
            if(value>que.back()){
                que.pop_back();
            }else{
                break;
            }
        }
        que.push_back(value);
    }
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int>res;
        for(int i=0;i<k;i++) push(nums[i]);
        res.push_back(que.front());
        for(int i=k;i<nums.size();i++){
            if(que.front()==nums[i-k]) que.pop_front();
            push(nums[i]);
            res.push_back(que.front());
        }   
        return res;
        
    }
};
```

## [**347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

```c++
class Solution {
public:
    class comparison{
        public:
            bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
            return lhs.second > rhs.second;
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_set<int> set;
        unordered_map<int,int> map;
        for(int i=0;i<nums.size();i++){
            map[nums[i]]++;
        }
        priority_queue<pair<int,int>,vector<pair<int,int>>,comparison> pri_que;
        
        for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
            pri_que.push(*it);
            if (pri_que.size() > k) {
                pri_que.pop();
            }
        }
        vector<int> result(k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pri_que.top().first;
            pri_que.pop();
        }
        return result;

    }
};
```




# 回溯

## [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用 一次 。

注意：解集不能包含重复的组合。

在判断重复那里卡住了，其实只要在回溯pop后一直i加到和当前数字不同就可以了。。

```c++
class Solution {
    void backtrack(vector<vector<int>>& res, vector<int> tmp, vector<int> candidates, int target, int sum, int start){
        if(sum==target){
            res.push_back(tmp);
            return;
        }
        
        for(int i=start;i<candidates.size();i++){
            int flag=0;
            if(sum>=target) break;
            tmp.push_back(candidates[i]);
            backtrack(res, tmp, candidates, target, sum+candidates[i], i+1);
            tmp.pop_back();
            while((i+1)<candidates.size()&&candidates[i]==candidates[i+1]){
                i++;
            }
        }
    }
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> tmp;
        sort(candidates.begin(),candidates.end());
        backtrack(res, tmp, candidates, target, 0, 0);
        return res;
    }
};
```

## [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

**回文串** 是正着读和反着读都一样的字符串。

ps:backtrack的判断函数想错了，应该是start开始长为i的下一个串是不是回文，一开始想的是记录pre，找到最长的回文，但是这样去重麻烦，且pop_back的时候应该是退一个字符，很麻烦。。。（太菜了）

```c++
class Solution {
    bool isCycle(string s){
        for(int i=0;i<s.size()/2;i++){
            if(s[i]!=s[s.size()-i-1]) return false;
        }
        return true;
    }
    void backtrack(vector<vector<string>>& res, vector<string>tmp, string s, string pre, int start){
        if(start>=s.size()){
            
            res.push_back(tmp);
        }
        for(int i=start;i<s.size();i++){
            string t = s.substr(start,i-start+1);
            if(isCycle(t)){
                tmp.push_back(t);
                //backtrack(res, tmp, s, pre, i+1);
            }else{
                continue;
            }
            backtrack(res,tmp,s,pre,i+1);
            tmp.pop_back();

        }
    }
public:
    vector<vector<string>> partition(string s) {
        vector<vector<string>> res;
        vector<string> tmp;
        backtrack(res, tmp, s, "", 0);
        return res;
    }
};
```

## [93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)(没做出来)···

感觉有点强行回溯，四层for循环应该能写出来，被回溯限制住了

```
class Solution {
private:
    vector<string> result;
    
    void backtracking(string& s, int startIndex, int pointNum) {
        if (pointNum == 3) { 
            
            if (isValid(s, startIndex, s.size() - 1)) {
                result.push_back(s);
            }
            return;
        }
        for (int i = startIndex; i < s.size(); i++) {
            if (isValid(s, startIndex, i)) { /
                s.insert(s.begin() + i + 1 , '.'); 
                pointNum++;
                backtracking(s, i + 2, pointNum);   
                pointNum--;                        
                s.erase(s.begin() + i + 1);        
            } else break;
        }
    }
   
    bool isValid(const string& s, int start, int end) {
        if (start > end) {
            return false;
        }
        if (s[start] == '0' && start != end) {
                return false;
        }
        int num = 0;
        for (int i = start; i <= end; i++) {
            if (s[i] > '9' || s[i] < '0') {
                return false;
            }
            num = num * 10 + (s[i] - '0');
            if (num > 255) { 
                return false;
            }
        }
        return true;
    }
public:
    vector<string> restoreIpAddresses(string s) {
        result.clear();
        if (s.size() < 4 || s.size() > 12) return result; // 算是剪枝了
        backtracking(s, 0, 0);
        return result;
    }
};
```

## [78. 子集](https://leetcode.cn/problems/subsets/)

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

```c++
class Solution {
    void backtrack(vector<vector<int>>& res, vector<int> tmp, int start, vector<int>nums){
        res.push_back(tmp);
        if(start==nums.size()){
            return;
        }
        for(int i=start;i<nums.size();i++){
            tmp.push_back(nums[i]);
            backtrack(res, tmp, i + 1, nums);
            tmp.pop_back();
        }
    }
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> tmp;
        backtrack(res, tmp, 0, nums);
        return res;
    }
};
```

## [491. 递增子序列](https://leetcode.cn/problems/non-decreasing-subsequences/)

给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。

数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。

ps:没想到怎么去重 唉，然后uset的位置还放错了。。

```c++
class Solution {
    void backtrack(vector<vector<int>>& res, vector<int> tmp, int start, vector<int> nums) {
        //if()
        unordered_set<int> uset;
        for (int i = start; i < nums.size(); i++) {
            
            for (int j = 0; i < start; j++) 
                uset.insert(nums[j]);
            if (uset.find(nums[i]) != uset.end()) continue;
            if (tmp.size() == 0) {
                uset.insert(nums[i]);
                tmp.push_back(nums[i]);
                backtrack(res, tmp, i + 1, nums);
                tmp.pop_back();
                while (i < nums.size() - 1 && nums[i] == nums[i + 1]) i++;
                continue;
            }

            if (nums[i] >= tmp.back()) {

                uset.insert(nums[i]);
                tmp.push_back(nums[i]);
                res.push_back(tmp);
                backtrack(res, tmp, i + 1, nums);
                tmp.pop_back();
                while (i < nums.size() - 1 && (nums[i] == nums[i + 1])) i++;
            }

        }
    }
public:
    vector<vector<int>> findSubsequences(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> tmp;
        //sort(nums.begin(),nums.end());
        backtrack(res, tmp, 0, nums);
        return res;
    }
};
```

## [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

给定一个可包含重复数字的序列 `nums` ，***按任意顺序*** 返回所有不重复的全排列。

```c++
class Solution {
    void backtrack(vector<vector<int>>& res, vector<int> tmp, vector<int> nums, int start){
        if(nums.size()==start) res.push_back(tmp);
        unordered_set<int> set;
        for(int i=start;i<nums.size();i++){
            //if(i<nums.size()-1&&nums[i]==nums[i+1]) continue;
            if(set.find(nums[i])!=set.end()) continue;
            set.insert(nums[i]);
            swap(nums[start],nums[i]);
            tmp.push_back(nums[start]);
            backtrack(res,tmp,nums,start+1);
            tmp.pop_back();
            swap(nums[start],nums[i]);
        }
    }
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> tmp;
        sort(nums.begin(),nums.end());
        backtrack(res, tmp, nums, 0);
        return res;
    }
};
```

## [332. 重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/)(困难，不会)

给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。

所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典排序返回最小的行程组合。

例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。
假定所有机票至少存在一种合理的行程。且所有的机票 必须都用一次 且 只能用一次。

ps：暴力法超时。。还是得用dfs, map是有序的

**`for(pair<const string, int>& target : targets[result[result.size() - 1]])`**

类似`python`的`for target,i in enumerate(targets):`

```c++
class Solution {
    unordered_map<int,int> map;
    void backtrack(vector<string> res, vector<vector<string>> &res1, vector<vector<string>>& tickets, string pre){
        if(res.size()==tickets.size()+1){
            res1.push_back(res);
        }
        for(int i=0;i<tickets.size();i++){
            if(tickets[i][0]==pre&&map[i]==0){
                //res.push_back(tickets[i][0]);
                res.push_back(tickets[i][1]);
                pre = tickets[i][1];
                map[i]++;
                backtrack(res,res1,tickets,pre);
                //res.pop_back();
                res.pop_back();
                pre = tickets[i][0];
                map[i]--;
            }
        }
    }
public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        vector<string> res = { "JFK" };
        vector<vector<string>> res1;
        for (int i = 0; i < tickets.size(); i++) map[i] = 0;
        backtrack(res, res1, tickets, "JFK");
        int minIndex = 0;
        bool flag = false;
        string comp;
        unordered_map<int,int> map1;
        for (int j = 0; j < res1.size(); j++) map1[j]=0;
        for (int i = 0; i < tickets.size(); i++) {
            comp = res1[minIndex][i];
            
            for (int j = 1; j < res1.size(); j++) {
                if(map1[j]>=1) continue;
                string tt = res1[j][i];
                //if(comp!=tt)flag = true;
                if (comp > tt) {
                    map1[minIndex]++;
                    minIndex = j;
                    for (int k = 0; k < j; k++) map1[k]++;
                    comp = res1[j][i];                 
                }else if (comp < tt) map1[j]++;
                
            }
            //if (flag) break;
        }
        return res1[minIndex];
    }
};
```

回溯正解

```
class Solution {
private:
// unordered_map<出发机场, map<到达机场, 航班次数>> targets
unordered_map<string, map<string, int>> targets;
bool backtracking(int ticketNum, vector<string>& result) {
    if (result.size() == ticketNum + 1) {
        return true;
    }
    for (pair<const string, int>& target : targets[result[result.size() - 1]]) {
        if (target.second > 0 ) { // 记录到达机场是否飞过了
            result.push_back(target.first);
            target.second--;
            if (backtracking(ticketNum, result)) return true;
            result.pop_back();
            target.second++;
        }
    }
    return false;
}
public:
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        targets.clear();
        vector<string> result;
        for (const vector<string>& vec : tickets) {
            targets[vec[0]][vec[1]]++; // 记录映射关系
        }
        result.push_back("JFK"); // 起始机场
        backtracking(tickets.size(), result);
        return result;
    }
};
```

## [37. 解数独](https://leetcode.cn/problems/sudoku-solver/)

问题出在没`return false`所以一直死循环了。。因为当前循环了1-9后找不到解，棋盘跳过了这个，下一次循环的时候这里还是没有解的，最后会变得剩下没有解的区域，所以 没解的时候要`return false`

```c++
#include<iostream>
#define day 7
#include<string>
#include<cmath>
#include<cstring>
#include <unordered_set>
#include <unordered_map>
#include<algorithm>
#include<queue>
#include<stack>
using namespace std;
class Solution {
    unordered_map<int, vector<char>> rows;
    unordered_map<int, vector<char>> cols;
    unordered_map<int, vector<char>> squ;

    bool isValid(vector<vector<char>>& board, int i, int j, char num) {
        if (find(rows[i].begin(), rows[i].end(), num) == rows[i].end())
            if (find(cols[j].begin(), cols[j].end(), num) == cols[j].end())
                if (find(squ[j / 3 + (i / 3) * 3].begin(), squ[j / 3 + (i / 3) * 3].end(), num) 
                    == squ[j / 3 + (i / 3) * 3].end())
                    return true;
        return false;
    }
    bool backtrack(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (int k = 0; k < 9; k++) {                     
                        if (isValid(board, i, j, '0' + k + 1)) {
                            char tm = '0' + k + 1;
                            board[i][j] = '0' + k + 1;
                            rows[i].push_back('0' + k + 1);
                            cols[j].push_back('0' + k + 1);
                            squ[j / 3 + (i / 3) * 3].push_back('0' + k + 1);
                            if(backtrack(board)) return true;
                            board[i][j] = '.';
                            rows[i].pop_back();
                            cols[j].pop_back();
                            squ[j / 3 + (i / 3) * 3].pop_back();
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }
public:
    void solveSudoku(vector<vector<char>>& board) {
        vector<int> row;
        vector<int> col;

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    rows[i].push_back(board[i][j]);
                    cols[j].push_back(board[i][j]);
                    int numSqu = j / 3 + (i / 3) * 3;//第几个
                    squ[numSqu].push_back(board[i][j]);
                }
            }
        }
        backtrack(board);
    }
};

```

# 贪心

## [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)

给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:

0 <= j <= nums[i] 
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

不难 但是错了好多次。。（状态？）

```
class Solution {
public:
    int jump(vector<int>& nums) {
        int next,maxj;
        int count=0;
        if(nums.size()==1) return 0;
        for(int i=0;i<nums.size();){
            next=i+nums[i];
            maxj=0;
            int nMax = nums[i]+i;
            //if(next>=nums.size()-1) break;
            for(int j=1;(j+i)<nums.size()&&j<=nums[i];j++){
                int tmp = i+j+nums[i+j];   
                 
                if(nMax<tmp||(i+j)==nums.size()-1){
                    next=i+j;
                    nMax = tmp;
                    //maxj=nums[j];
                }
            }
            //if(maxj>0)
            count++;
            i=next;
            if(next>=nums.size()-1) break;
            if(nums[next]==0) return -1;
        }
        return count;
    }
};
```

#### [1005. K 次取反后最大化的数组和](https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/)

```c++
class Solution {
public:
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        sort(nums.begin(),nums.end());
        while(k>0){
            nums[0] = -nums[0];
            k--;
            sort(nums.begin(),nums.end());
        }
        int result=0;
        for(int a:nums) result+=a;
        return result;
    }
};
//将A按绝对值进行排列，然后for从1到A.size每次就都是转换绝对值最大的负数，然后剩余的k是奇数就换最后一个（最小正数），是偶数就不用换（因为每次都换最小数）
class Solution {
static bool cmp(int a, int b) {
    return abs(a) > abs(b);
}
public:
    int largestSumAfterKNegations(vector<int>& A, int K) {
        sort(A.begin(), A.end(), cmp);       // 第一步
        for (int i = 0; i < A.size(); i++) { // 第二步
            if (A[i] < 0 && K > 0) {
                A[i] *= -1;
                K--;
            }
        }
        if (K % 2 == 1) A[A.size() - 1] *= -1; // 第三步
        int result = 0;
        for (int a : A) result += a;        // 第四步
        return result;
    }
};
```

## [134. 加油站](https://leetcode.cn/problems/gas-station/)

在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

给定两个整数数组 gas 和 cost ，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。

```c++
//暴力法：超时
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        
        for(int i=0;i<gas.size();i++){
            int rest = gas[i]-cost[i];
            if(rest<0) continue;
            int index = (i+1)%gas.size();
            while(rest>0&&index!=i){
                rest=rest+gas[index]-cost[index];
                index = (index+1)%gas.size();
            }
            if(rest>=0&&index == i) return i;
        }
        return -1;
    }
};
//O(n)的关键在于相通如果sum(gas-cost)>0那一定是有一个解的，且解就在最后一个sum<0的序列的下一个
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int start=0,sum1=0;
        for(int i=0;i<gas.size();i++) sum1+=gas[i] - cost[i];
        if(sum1<0) return -1;
        sum1=0;
        for(int i=0;i<gas.size();i++){
            sum1+=gas[i] - cost[i];

            if(sum1<0){
                start = i+1;
                sum1=0;
            }
        }
        return start;
    }
};
```

## [860. 柠檬水找零](https://leetcode.cn/problems/lemonade-change/)

```c++
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        unordered_map<int,int> map;//当前5 10的数量
        map[5]=0;
        map[10]=0;
        for(int i=0;i<bills.size();i++){
            if(bills[i]==5){
                map[5]++;
            }else if(bills[i]==10){
                map[10]++;
                map[5]--;
            }else if(bills[i]==20){
                int rest = 15;//
                map[20]++;
                while(rest>0){
                    if(map[10]>0&&rest>10){
                        map[10]--;
                        rest-=10;
                    }else{
                        map[5]--;
                        rest-=5;
                    }
                }
            }
            if(map[5]<0) return false;
        }
        return true;
        
    }
};
```
## [455. 分发饼干](https://leetcode.cn/problems/assign-cookies/)

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大
```
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(),g.end());//小孩
        sort(s.begin(),s.end());//饼干
        int assigned = 0;
        for(int i=0;i<s.size();i++){
            if(s[i]>=g[assigned]){
                assigned+=1;
            }
            if(assigned==g.size()) break;
        }
        return assigned;
    }
};
```

## [376. 摆动序列](https://leetcode.cn/problems/wiggle-subsequence/)

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。

相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。

```c++
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int res=1;
        int flag=0,flag2=0;
        int first=1;
        if(nums.size()==1) return 1;
        // if(nums[1]-nums[0]<0) flag=1;
        // else if(nums[1]-nums[0]>0) flag=-1;
        for(int i=1;i<nums.size();i++){
            if(first==1&&nums[i]!=nums[i-1]){
                if(nums[i]-nums[i-1]<0) flag=1;
                else if(nums[i]-nums[i-1]>0) flag=-1;
                first=0;
            }
            if((nums[i]-nums[i-1]<0&&flag==1)||nums[i]-nums[i-1]>0&&flag==-1)
                res++;
            if(nums[i]-nums[i-1]<0) flag=-1;
            else if(nums[i]-nums[i-1]>0) flag=1;   
        }
        return res;
    }
};
//动态规划
class Solution {
public:
    int dp[1005][2];
    int wiggleMaxLength(vector<int>& nums) {
        memset(dp, 0, sizeof dp);
        dp[0][0] = dp[0][1] = 1;
        for (int i = 1; i < nums.size(); ++i) {
            dp[i][0] = dp[i][1] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[j] > nums[i]) dp[i][1] = max(dp[i][1], dp[j][0] + 1);
            }
            for (int j = 0; j < i; ++j) {
                if (nums[j] < nums[i]) dp[i][0] = max(dp[i][0], dp[j][1] + 1);
            }
        }
        return max(dp[nums.size() - 1][0], dp[nums.size() - 1][1]);
    }
};
```

## [406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/)（没做出来）

假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

```c++
class Solution {
    static bool comp(vector<int> a, vector<int> b){
        if(a[0]==b[0]) return a[1]<b[1];
        return a[0]>b[0];
    }
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(),people.end(),comp);
        vector<vector<int>> res;
        for(int i=0;i<people.size();i++){
            res.insert(res.begin()+people[i][1],people[i]);
        }
        return res;
    }
};
```

## [452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)*(贪心是有点难。。)

在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

PS:这里不用static bool comp(vector<int>& a,vector<int>& b)会超时（可能是因为不引用的话会新建变量？比较耗时）

```c++
class Solution {
    static bool comp(vector<int>& a,vector<int>& b){
        //if(a[0]==b[0]) return a[1]<b[1];
        return a[0]<b[0];
    }
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if (points.size() == 0) return 0;
        sort(points.begin(),points.end(),comp);
        int result=1;
        for(int i=1;i<points.size();i++){
            if(points[i][0]>points[i-1][1]) result++;
            else{
                points[i][1] = min(points[i-1][1],points[i][1]);
            }
        }
        return result;
    }
};
```

## [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/)

给定一个区间的集合 intervals ，其中 intervals[i] = [starti, endi] 。返回 需要移除区间的最小数量，使剩余区间互不重叠 。

贪心策略：按起点排序，重叠的时候选择结尾最小的
