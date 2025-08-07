use cylon::{Cylon, cylon_proto::cylon_api_server::CylonApiServer};
use cylon_config::CylonConfig;
use tonic::transport::Server;
use utils::init_logging;

#[allow(unused_imports)]
use tracing::{info, debug, error, warn};

mod utils;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CylonConfig::new()?;

    init_logging(config.debug);

    info!("Starting Cylon Engine");

    info!("Loading model and creating engine");
    let cylon = Cylon::new(&config)?;

    let addr = format!("{}:{}", config.listen_address, config.listen_port).parse()?;
    info!("Server listening: {}", addr);

    Server::builder()
        .add_service(CylonApiServer::new(cylon))
        .serve(addr)
        .await?;

    Ok(())
}