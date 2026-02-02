use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum TaskCommand {
    /// Create a new task.
    Create {
        name: String,
        description: String,
        /// Optional parent task names
        #[arg(long)]
        depends_on: Vec<String>,
    },
    /// List all tasks and their status.
    List,
    /// Mark a task as active/started.
    Start {
        name: String,
    },
    /// Mark a task as completed.
    Finish {
        name: String,
    },
}
