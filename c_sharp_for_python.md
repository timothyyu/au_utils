# C\# For Python Programmers

## Syntax and core concepts

### Basic Syntax

- Single-line comments are started with `//`. Multi-line comments are started with `/*` and ended with `*/`.

- C# uses braces (`{` and `}`) instead of indentation to organize code into blocks.
  If a block is a single line, the braces can be omitted. For example,

        if (foo) {
            bar();
        }

  can be shortened to

        if (foo)
            bar();

  However, avoid

        if (foo) {
            bar();
            baz();
        }
        else // matching if uses braces, but this does not
            somethingCompletelyDifferent();

  preferring

        if (foo) {
            bar();
            baz();
        }
        else { // matching if uses braces, so we do here even though it's one line
            somethingCompletelyDifferent();
        }

- Brace style is contested. The default C# style is to have braces on their own line for everything:

        class Thing
        {
            void foo()
            {
                if (a)
                {
                    bar();
                    baz();
                }
                else
                {
                    otherThing();
                }
            }
        }

   The author is of the opinion that this wastes vertical screen space and that it is easier to code when you can see
   more on your screen, and prefers [Stroustrup Style](http://en.wikipedia.org/wiki/Indent_style#Variant:_Stroustrup):

        class Thing {
            void foo()
            {
                if (a) {
                    bar();
                    baz();
                }
                else {
                    otherThing();
                }
            }
        }

   or even Java style:

        class Thing {
            void foo() {
                if (a) {
                    bar();
                    baz();
                } else {
                    otherThing();
                }
            }
        }

   All of these styles can be set in Visual Studio in Tools -> Options -> Text Editor -> C# -> Formatting -> New Lines

- Statements are ended with semicolons. A statement can therefore span multiple lines without any problem.
  This can be useful in some cases:

        someVariable += SomeReallyLongFunctionName(longParameterOne,
                                                   longParameterTwo);

- C# is a strongly-typed language. Unlike Python, variables must be explicitly declared before they are used,
  and you must specify their type when you declare them. Declarations take the form

        <type> <variable name>;

  or, to declare and initialize them in one statement,

        <type> <variable name> = <value>;

  This type cannot be changed. For example, if you declare an integer with

        int i = 42;

  you cannot later assign a string value, such as

        i = "I'm a string now!";

  Thankfully, when you declare and initialize a variable in the same statement, the compiler can determine the type
  from the right hand side of the assignment if you use the keyword `var`. For example,

        int port = 42;
        TcpListener listener = new TcpListener(port);

  can be shortened to

        var port = 42;
        var listener = new TcpListener(port);

  This may seem more natural to someone with experience in "weak-typed" languages like Python.

- Because of this strong typing, data structures (such as lists, queues, trees, etc.) must be told what type they will
  be containing using the following syntax:

        var intList = new List<int>(); // Declares a list of ints
        var dict = new Dictionary<int, string>(); // Declares a dictionary that maps ints to strings
        // and so on...

- For similar reasons, data structures cannot contain a mix of unrelated types. You cannot have a `List` that contians
  some mix of strings and integers for example. You can, of course, make a list of some base class and add instances
  of derived classes. Assuming `Cat` and `Dog` are derived from the `Animal` class, the following code is valid:

        var petList = new List<Animal>();
        petList.add(new Cat());
        petList.add(new Dog());

  If you are looking to contain a mix of types for a specific purpose, see the section below on user-defined types.

- Arrays are fixed-size (you cannot append to them, use `List` for such cases) and take the following syntax:

        var firstTen = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; // An array of the first ten positive integers

        // .Length will retrieve an array's length, similar to len() in Python
        if (firstTen.Length != 10)
            throw new Exception("Reality is not working properly."); // Throws an exception (see below)

        // Create an array of 20 integers, initialized to their default value, i.e., 0
        var empty = new int[20];

### Control Flow:

- `if x = 0:` becomes `if (x == 0)`. Note that in C#, `==` is the comparison operator.
   A single equals is used for assignment, and the compiler will not allow it inside a conditional like `if`.

- `++i` is shorthand for `i = i + 1`. Placing `++` before or after the variable (e.g. `i++`)
  [can have different effects in some situations](http://stackoverflow.com/a/4706239/713961).

- `for i in range(0, 10)` becomes `for (int i = 0; i < 10; ++i)`.
  This takes the form  
  `for (<declaration>; <continue looping while this is true>; <at the end of each iteration>)`.  
  So, we're declaring a counter variable `i`, initializing it to 0, looping until it is equal to 10,
  and incrementing it by one each time.

- `for item in myArray` becomes `foreach (var item in myArray)`

- Exception handling:

        try:
            ...
        except SomeError as e:
            ...

  becomes

        try {
            ...
        }
        catch (SomeError e) {
            ...
        }

- Throwing exceptions: `raise Exception("It broke")` becomes `throw new Exception("It broke");`.

### Boolean expressions

- Boolean values `True` and `False` are lowercase in C# (`true` and `false`)

- C# uses symbols instead of keywords for logical operators.
  - `and` is `&&`
  - `or` is `||`
  - `not` is `!`

- The [bitwise](http://en.wikipedia.org/wiki/Bitwise_operation) versions of these operators are `&`, `|`, and `~`,
  respectively.

## User-Defined datatypes

- Like Python, C# allows the user to define their own classes with variables and methods.

  In Python:

        class MyClass(BaseClass): # Inherit BaseClass

             def __init__(self, foo):
                # Initialize the class with the help of the argument foo

             # Other contents

        # Elsewhere...
        # Instantiate an instance of MyClass
        instance = MyClass(42)

  In C#:

        class MyClass : BaseClass { // Inherit BaseClass

            public MyClass(int foo)
            {
                // Initialize the class with the help of the argument foo
            }

            // Other contents
        }

        // Elsewhere...
        // Instantiate an instance of MyClass
        var instance = new MyClass(42);

- In Python, member [methods](http://stackoverflow.com/q/70528/713961)
  and [variables](http://stackoverflow.com/q/1641219/713961) can be made "private" to the class using an `__` prefix,
  but this just mangles their names. Nothing keeps the programmer from actually using them outside the class,
  who is trusted to not do this. C# has less faith in humanity and, as a consequence, provides the following
  accessibility levels:
  - **public** - All code can access this.
  - **internal** - Only other code in this assembly (i.e. program or library) can access this.
    This can be useful if you are making a library and want to make classes or methods inaccessible to the end-user.
  - **protected** - Only other code in this class _and_ derived classes can access this.
  - **protected internal** - Only other code in this class _and_ derived classes in this assembly can access this.
  - **private** - Only other code in this class can access this.

- In C#, you cannot have free-standing, "global" functions and variables like you can in Python. Because of this,
  _static_ members, which can be accessed without creating an instance of the class. Python has
  [them as well](http://stackoverflow.com/a/735978/713961), but they're not used as often since it allows global
  functions and variables.

- C# has a feature called [properties](http://msdn.microsoft.com/en-us/library/x9fsa0sw.aspx),
  which allow you to replace instances where you would use `getX` and `setX` style functions with a field that
  looks like a variable but calls the proper "getter" and "setter" methods when the value is accessed or modified:

        private int foo; // Backing member variable
        public int Foo // Properties are usually capitalized camel case
        {
            get // Called when Foo is accessed
            {
                // Any logic or updating of related state here
                // ...

                return foo;
            }
            Set // Called when Foo is changed
            {
                // Any logic or updating of related state here
                // ...

                foo = value;
            }
        }

  If your `get` and `set` methods do nothing but use the backing variable,
  C# can auto-implement the property and backing variable:

        public int Foo { get; set; }

  This is preferred over public member variables, as you can later modify the property if you need it to do more
  without changing any code that uses the property. The getter and setter can also be given different access levels:

        // A property that can only be modified inside the class that contains it:
        public int Foo { get; private set; }

- While Python variables [use reference-like bahavior](http://stackoverflow.com/q/6158907/713961),
  C# has _value types_ as well as reference types. Value types act as individual copies of whatever data they contain,
  whereas reference types _point_ to instances of the data. Classes are reference types.
  The programmer may also create their own value types using the `struct` keyword.
  A trivial example of all of this is below:

        // This code is not meant to be useful,
        // but just to explain the difference between value and reference types.

        class RefType {
            public int contents;

            // Trivial constructor to allow us to initialize contents
            public RefType(int c) { contents = c; }
        }

        struct ValType {
            public int contents;

            // Trivial constructor to allow us to initialize contents
            public ValType(int c) { contents = c; }
        }

        // Elsewhere...
        ValType val1 = new ValType(42);
        ValType val2 = val1;
        val2.contents = 20;
        // val1.contents is 42 and val2.contents is 20
        // since value types act as individual copies of whatever data they contain.

        RefType ref1 = new RefType(30);
        RefType ref2 = ref1;
        ref2.contnets = 1;
        // ref1.contents is 1 and ref2.contents is 1 because both references point at the same object.

        ref2 = null; // References can refer to nothing. null is the C# equivalent of "None" in Python.

  If this doesn't help, there are [many](http://www.albahari.com/valuevsreftypes.aspx)
  [online](http://yoda.arachsys.com/csharp/references.html)
  [articles](http://blogs.msdn.com/b/ericlippert/archive/2009/04/27/the-stack-is-an-implementation-detail.aspx)
  that explain the difference.

## Resource handling

Since C# is a [garbage collected](http://msdn.microsoft.com/en-us/library/ms973837.aspx) language,
memory will be automatically managed by the .NET runtime. Other resources, however, such as file handles,
network sockets, etc. must be closed manually by the programmer. This is handled by the
[IDisposable interface](http://msdn.microsoft.com/en-us/library/system.idisposable(v=vs.110\).aspx).
Classes that implement this interface have a `Dispose` method, which can be called to close resources.
C# also provide the [`using` statement](http://msdn.microsoft.com/en-us/library/yh598w02.aspx),
which works similarly to [Python's `with`](http://docs.python.org/2/reference/compound_stmts.html#with) statement.
The following two blocks of are equivalent:

    using (Font font1 = new Font("Arial", 10.0f))
    {
        byte charset = font1.GdiCharSet;
    }

is equivalent to

    {
        Font font1 = new Font("Arial", 10.0f);
        try
        {
            byte charset = font1.GdiCharSet;
        }
        finally
        {
            if (font1 != null)
              ((IDisposable)font1).Dispose();
        }
    }

## Synchronization

- Unlike most Python implementations, C# does not have a
  [Global Interpreter Lock](http://en.wikipedia.org/wiki/Global_Interpreter_Lock).
  This allows it to run several threads in parallel, on different CPU cores.

- C# has locks ([Python's equivalent here](http://docs.python.org/2/library/threading.html#lock-objects))
  built in on the language level. Any reference type (i.e. class) can be used as a re-entrant lock using the `lock`
  keyword:

        class Account {
            decimal balance;
            private Object thisLock = new Object();

            public void Withdraw(decimal amount)
            {
                // Everything in this block is a critical section protected by thisLock
                lock (thisLock) {
                    if (amount > balance) {
                        throw new Exception("Insufficient funds");
                    }
                        balance -= amount;
                }
            }
        }

  See [the MSDN page](http://msdn.microsoft.com/en-us/library/c5kehkcz.aspx) for recommended guidelines.

- C# also comes with your usual bag of tools for synchronization, such as monitors, semaphores, etc.
  See [this](http://msdn.microsoft.com/en-us/library/ms173179.aspx).

-  Reads and writes to `bool`, `char`, `byte`, `sbyte`, `short`, `ushort`, `uint`, `int`, `float`,
   and reference types are
   [guaranteed to be atomic by the C# language spec](http://msdn.microsoft.com/en-us/library/aa691278%28VS.71%29.aspx).
   This means that you can, for example, set a `bool` in one thread to indicate to another thread
   that it should exit a loop without the use of a lock. This **does not**, however, mean that read-modify-write
   operations (such as incrementing a variable or checking it, then setting it) are atomic. Also, to be used
   across multiple threads in the manner described above, these variables must be declared with the
   [`volatile` keyword](http://msdn.microsoft.com/en-us/library/x13ttww7.aspx), which forbids the compiler from
   caching a copy of the variable in a CPU core's registers.
   The link on `volatile` above also includes an example of the "boolean as an exit flag" approach mentioned above.
